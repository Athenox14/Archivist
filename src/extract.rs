//! Extraction de texte (.txt/.md/.pdf) et découpage en chunks.

use anyhow::Result;
use std::path::Path;

/// Quantité max de texte embeddée par fichier. Au-delà = inutile pour la
/// recherche et dangereux en RAM (ex. CSV de 1 Go). On ne lit que ce début.
pub const TEXT_MAX: usize = 1024 * 1024; // 1 Mo

/// Lit au plus `max` octets d'un fichier → String (lossy).
pub fn read_capped(path: &Path, max: usize) -> Result<String> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)?
        .take(max as u64)
        .read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Tronque une String à `max` octets, sur une frontière de caractère.
pub fn truncate_text(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
    }
    s
}

/// Extrait le texte brut d'un fichier supporté (plafonné à `TEXT_MAX`).
pub fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" => read_capped(path, TEXT_MAX),
        // pdf-extract peut PANIQUER sur certains PDF → on isole le panic pour
        // ne pas tuer l'indexation ; un PDF illisible est simplement sauté.
        "pdf" => Ok(truncate_text(extract_pdf_safe(path), TEXT_MAX)),
        _ => Ok(String::new()),
    }
}

/// Extraction PDF EN MÉMOIRE (appelée dans le sous-processus dédié).
/// `catch_unwind` couvre les panics simples ; un double-panic/abort fera
/// planter ce processus enfant — sans toucher au parent.
pub fn extract_pdf_raw(path: &Path) -> String {
    let p = path.to_path_buf();
    std::panic::catch_unwind(|| pdf_extract::extract_text(&p).unwrap_or_default())
        .unwrap_or_default()
}

/// Extraction PDF ROBUSTE : déléguée à un SOUS-PROCESSUS `archivist extract-pdf`.
/// pdf-extract pouvant *abort* (incatchable en mémoire), l'isoler dans un enfant
/// garantit que le crash ne tue jamais l'indexation. Repli en mémoire si le
/// binaire `archivist` est introuvable.
fn extract_pdf_safe(path: &Path) -> String {
    match archivist_bin() {
        Some(bin) => {
            let mut cmd = std::process::Command::new(bin);
            cmd.arg("extract-pdf").arg("--path").arg(path);
            no_window(&mut cmd);
            match cmd.output() {
                Ok(out) if out.status.success() => {
                    String::from_utf8_lossy(&out.stdout).into_owned()
                }
                _ => {
                    log::warn!(
                        "extraction pdf échouée/crashée (sautée) : {}",
                        path.display()
                    );
                    String::new()
                }
            }
        }
        None => extract_pdf_raw(path), // repli best-effort
    }
}

/// Localise le binaire `archivist` (à côté de l'exe courant, ou l'exe lui-même).
fn archivist_bin() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    if exe.file_name()?.to_str()?.starts_with("archivist") {
        return Some(exe);
    }
    let name = if cfg!(windows) {
        "archivist.exe"
    } else {
        "archivist"
    };
    let cand = exe.parent()?.join(name);
    cand.exists().then_some(cand)
}

#[cfg(windows)]
fn no_window(cmd: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
}
#[cfg(not(windows))]
fn no_window(_cmd: &mut std::process::Command) {}

/// Découpe en chunks par paragraphe (séparés par lignes vides), en
/// fusionnant les courts pour viser ~`target_chars`. Filtre le vide.
pub fn chunk_text(text: &str, target_chars: usize) -> Vec<String> {
    const MAX_CHUNKS: usize = 300; // borne le nb d'embeddings par fichier
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut push = |chunks: &mut Vec<String>, cur: &mut String| {
        if !cur.trim().is_empty() {
            chunks.push(std::mem::take(cur));
        }
    };
    'outer: for para in text.split("\n\n") {
        // découpe les paragraphes géants (ex. CSV mono-ligne) en tranches
        let mut rest = para.trim();
        while !rest.is_empty() {
            let mut take = target_chars.min(rest.len());
            while take < rest.len() && !rest.is_char_boundary(take) {
                take += 1;
            }
            let piece = &rest[..take];
            rest = &rest[take..];
            if cur.len() + piece.len() + 1 > target_chars && !cur.is_empty() {
                push(&mut chunks, &mut cur);
                if chunks.len() >= MAX_CHUNKS {
                    break 'outer;
                }
            }
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(piece);
            if cur.len() >= target_chars {
                push(&mut chunks, &mut cur);
                if chunks.len() >= MAX_CHUNKS {
                    break 'outer;
                }
            }
        }
    }
    push(&mut chunks, &mut cur);
    chunks
}
