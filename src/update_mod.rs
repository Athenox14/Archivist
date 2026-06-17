//! Mise à jour automatique depuis les releases GitHub (utilisé par le GUI).

pub const REPO_OWNER: &str = "Athenox14";
pub const REPO_NAME: &str = "Archivist";

/// Cherche une version plus récente, l'installe, renvoie `true` si mise à jour
/// appliquée (l'appelant doit redémarrer). Toute erreur est silencieuse.
pub fn try_update() -> bool {
    if std::env::var("ARCHIVIST_NO_UPDATE").is_ok() {
        return false;
    }
    // self_update peut paniquer (pas de release, réseau, TLS…) → on isole pour
    // ne jamais empêcher le logiciel de démarrer.
    std::panic::catch_unwind(|| {
        let res = self_update::backends::github::Update::configure()
            .repo_owner(REPO_OWNER)
            .repo_name(REPO_NAME)
            .bin_name("gui")
            .current_version(self_update::cargo_crate_version!())
            .no_confirm(true)
            .show_download_progress(false)
            .build()
            .and_then(|u| u.update());
        matches!(res, Ok(s) if s.updated())
    })
    .unwrap_or(false)
}

/// Relance l'exécutable courant puis quitte (après mise à jour).
pub fn restart() -> ! {
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::process::Command::new(exe).spawn();
    }
    std::process::exit(0);
}
