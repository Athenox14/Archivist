//! Couche SQLite — recherche UNIQUEMENT. Chemins RELATIFS pour portabilité.
//!
//! Schéma :
//!   files(id, rel_path, hash, size, mtime)
//!   image_embeddings(file_id, vec BLOB)        -- f32[]
//!   text_chunks(file_id, chunk_idx, text, vec BLOB)

use anyhow::Result;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::path::Path;

pub struct Db {
    pub conn: Connection,
}

/// Encode un &[f32] en blob little-endian.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Décode un blob en Vec<f32>.
pub fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let db = Db { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Ouverture LECTURE SEULE : ne crée rien, n'initialise pas le schéma.
    /// Pour l'état/la recherche — évite de créer une DB vide par erreur.
    pub fn open_ro(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Db { conn })
    }

    /// Tailles distinctes déjà connues (pour décider quoi hasher cross-source).
    pub fn distinct_sizes(&self) -> Result<std::collections::HashSet<u64>> {
        let mut stmt = self.conn.prepare("SELECT DISTINCT size FROM files")?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        Ok(rows.filter_map(|r| r.ok()).map(|s| s as u64).collect())
    }

    /// Cherche un fichier DÉJÀ archivé, de même hash, mais hors du préfixe donné
    /// (= autre source). Sert à la dédup cross-source (hardlink au lieu de
    /// recompresser). `not_prefix` est de la forme "label/".
    pub fn find_other_with_hash(&self, hash: &str, not_prefix: &str) -> Result<Option<String>> {
        let like = format!("{not_prefix}%");
        Ok(self
            .conn
            .query_row(
                "SELECT rel_path FROM files WHERE hash=?1 AND rel_path NOT LIKE ?2 \
                 ORDER BY rel_path LIMIT 1",
                params![hash, like],
                |r| r.get::<_, String>(0),
            )
            .optional()?)
    }

    /// Fichiers avec un hash de contenu réel (pour la vérif d'intégrité).
    pub fn all_files_with_hash(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT rel_path, hash FROM files WHERE hash IS NOT NULL AND hash NOT LIKE 'u:%'",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Tous les chemins relatifs (pour la recherche par nom de fichier).
    pub fn all_rel_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT rel_path FROM files")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id       INTEGER PRIMARY KEY,
                rel_path TEXT NOT NULL UNIQUE,
                hash     TEXT,
                size     INTEGER NOT NULL,
                mtime    INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS image_embeddings (
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                vec     BLOB NOT NULL,
                PRIMARY KEY (file_id)
            );
            CREATE TABLE IF NOT EXISTS text_chunks (
                file_id   INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_idx INTEGER NOT NULL,
                text      TEXT NOT NULL,
                vec       BLOB NOT NULL,
                PRIMARY KEY (file_id, chunk_idx)
            );
            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
            "#,
        )?;
        Ok(())
    }

    /// Upsert un fichier par rel_path. Renvoie (id, besoin_reindex).
    /// besoin_reindex = true si nouveau OU mtime/size changé.
    pub fn upsert_file(
        &self,
        rel_path: &str,
        hash: Option<&str>,
        size: u64,
        mtime: i64,
    ) -> Result<(i64, bool)> {
        let existing: Option<(i64, i64, i64)> = self
            .conn
            .query_row(
                "SELECT id, size, mtime FROM files WHERE rel_path = ?1",
                params![rel_path],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        match existing {
            Some((id, old_size, old_mtime)) => {
                let changed = old_size != size as i64 || old_mtime != mtime;
                if changed {
                    self.conn.execute(
                        "UPDATE files SET hash=?1, size=?2, mtime=?3 WHERE id=?4",
                        params![hash, size as i64, mtime, id],
                    )?;
                    // contenu changé → embeddings périmés
                    self.conn
                        .execute("DELETE FROM image_embeddings WHERE file_id=?1", params![id])?;
                    self.conn
                        .execute("DELETE FROM text_chunks WHERE file_id=?1", params![id])?;
                }
                Ok((id, changed))
            }
            None => {
                self.conn.execute(
                    "INSERT INTO files (rel_path, hash, size, mtime) VALUES (?1,?2,?3,?4)",
                    params![rel_path, hash, size as i64, mtime],
                )?;
                Ok((self.conn.last_insert_rowid(), true))
            }
        }
    }

    pub fn has_image_embedding(&self, file_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM image_embeddings WHERE file_id=?1",
            params![file_id],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    pub fn has_text_chunks(&self, file_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM text_chunks WHERE file_id=?1",
            params![file_id],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    pub fn insert_image_embedding(&self, file_id: i64, vec: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO image_embeddings (file_id, vec) VALUES (?1, ?2)",
            params![file_id, vec_to_blob(vec)],
        )?;
        Ok(())
    }

    pub fn insert_text_chunk(
        &self,
        file_id: i64,
        chunk_idx: usize,
        text: &str,
        vec: &[f32],
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO text_chunks (file_id, chunk_idx, text, vec) VALUES (?1,?2,?3,?4)",
            params![file_id, chunk_idx as i64, text, vec_to_blob(vec)],
        )?;
        Ok(())
    }

    /// Compteurs pour l'affichage d'état : (fichiers, embeddings image, chunks texte).
    pub fn counts(&self) -> Result<(i64, i64, i64)> {
        let files: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let imgs: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM image_embeddings", [], |r| r.get(0))?;
        let chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM text_chunks", [], |r| r.get(0))?;
        Ok((files, imgs, chunks))
    }

    /// Tous les embeddings image : (rel_path, vec).
    pub fn all_image_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.rel_path, e.vec FROM image_embeddings e JOIN files f ON f.id=e.file_id",
        )?;
        let rows = stmt.query_map([], |r| {
            let path: String = r.get(0)?;
            let blob: Vec<u8> = r.get(1)?;
            Ok((path, blob_to_vec(&blob)))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Tous les chunks texte : (rel_path, chunk_idx, text, vec).
    pub fn all_text_chunks(&self) -> Result<Vec<(String, i64, String, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.rel_path, c.chunk_idx, c.text, c.vec \
             FROM text_chunks c JOIN files f ON f.id=c.file_id",
        )?;
        let rows = stmt.query_map([], |r| {
            let path: String = r.get(0)?;
            let idx: i64 = r.get(1)?;
            let text: String = r.get(2)?;
            let blob: Vec<u8> = r.get(3)?;
            Ok((path, idx, text, blob_to_vec(&blob)))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}
