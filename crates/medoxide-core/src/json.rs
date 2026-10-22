//! Écriture minimale de JSON, pour les fichiers `.json` des dérivées BIDS (`dataset_description.json`, fichiers voisins des images et des tableaux).
//!
//! Pas de lecture, et pas de dépendance : les fichiers écrits ici sont petits, et leur forme est fixée par le code qui les écrit.

/// Une valeur JSON. Un objet garde l'ordre de ses clés (celui dans lequel on les écrit).
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` ou `false`.
    Bool(bool),
    /// Nombre réel ; un nombre non fini s'écrit `null` (JSON n'a ni NaN ni infini).
    Nombre(f64),
    /// Nombre entier.
    Entier(i64),
    /// Chaîne de caractères (échappée à l'écriture).
    Texte(String),
    /// Liste de valeurs.
    Liste(Vec<Json>),
    /// Objet : paires (clé, valeur), dans l'ordre d'écriture.
    Objet(Vec<(String, Json)>),
}

impl Json {
    /// Chaîne de caractères à partir de tout ce qui se convertit en `String`.
    pub fn texte(valeur: impl Into<String>) -> Json {
        Json::Texte(valeur.into())
    }

    /// Objet à partir de paires (clé, valeur).
    pub fn objet<'a>(paires: impl IntoIterator<Item = (&'a str, Json)>) -> Json {
        Json::Objet(paires.into_iter().map(|(cle, v)| (cle.to_string(), v)).collect())
    }

    /// Texte JSON, indenté de deux espaces par niveau et terminé par un saut de ligne.
    pub fn en_texte(&self) -> String {
        let mut sortie = String::new();
        self.ecrire(&mut sortie, 0);
        sortie.push('\n');
        sortie
    }

    fn ecrire(&self, sortie: &mut String, niveau: usize) {
        let marge = |n: usize| "  ".repeat(n);
        match self {
            Json::Null => sortie.push_str("null"),
            Json::Bool(b) => sortie.push_str(if *b { "true" } else { "false" }),
            // `{:?}` donne la plus courte représentation qui se relit exactement ; elle est valide en JSON (`1.0`, `1e-5`).
            Json::Nombre(x) if x.is_finite() => sortie.push_str(&format!("{x:?}")),
            Json::Nombre(_) => sortie.push_str("null"),
            Json::Entier(n) => sortie.push_str(&n.to_string()),
            Json::Texte(t) => echapper(t, sortie),
            Json::Liste(v) if v.is_empty() => sortie.push_str("[]"),
            Json::Liste(v) => {
                sortie.push_str("[\n");
                for (i, element) in v.iter().enumerate() {
                    sortie.push_str(&marge(niveau + 1));
                    element.ecrire(sortie, niveau + 1);
                    sortie.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                sortie.push_str(&marge(niveau));
                sortie.push(']');
            }
            Json::Objet(v) if v.is_empty() => sortie.push_str("{}"),
            Json::Objet(v) => {
                sortie.push_str("{\n");
                for (i, (cle, valeur)) in v.iter().enumerate() {
                    sortie.push_str(&marge(niveau + 1));
                    echapper(cle, sortie);
                    sortie.push_str(": ");
                    valeur.ecrire(sortie, niveau + 1);
                    sortie.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                sortie.push_str(&marge(niveau));
                sortie.push('}');
            }
        }
    }
}

/// Ajoute `texte` entre guillemets, avec les échappements de JSON (`"`, `\`, caractères de contrôle).
fn echapper(texte: &str, sortie: &mut String) {
    sortie.push('"');
    for c in texte.chars() {
        match c {
            '"' => sortie.push_str("\\\""),
            '\\' => sortie.push_str("\\\\"),
            '\n' => sortie.push_str("\\n"),
            '\r' => sortie.push_str("\\r"),
            '\t' => sortie.push_str("\\t"),
            c if (c as u32) < 0x20 => sortie.push_str(&format!("\\u{:04x}", c as u32)),
            c => sortie.push(c),
        }
    }
    sortie.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forme de l'écriture : indentation de deux espaces, clés dans l'ordre donné, listes et objets vides sur une ligne, saut de ligne final.
    #[test]
    fn the_text_is_indented_and_keeps_the_key_order() {
        let j = Json::objet([
            ("b", Json::Entier(2)),
            ("a", Json::Liste(vec![Json::Bool(true), Json::Null])),
            ("v", Json::Liste(vec![])),
            ("o", Json::Objet(vec![])),
            ("x", Json::Nombre(0.8)),
        ]);
        assert_eq!(
            j.en_texte(),
            "{\n  \"b\": 2,\n  \"a\": [\n    true,\n    null\n  ],\n  \"v\": [],\n  \"o\": {},\n  \"x\": 0.8\n}\n"
        );
    }

    /// Guillemets, antislash, retours à la ligne et caractères de contrôle sont échappés (une ligne de commande en contient) ; les accents restent tels quels.
    #[test]
    fn the_strings_are_escaped() {
        let t = Json::texte("a\"b\\c\nd\te\u{1}é").en_texte();
        assert_eq!(t, "\"a\\\"b\\\\c\\nd\\te\\u0001é\"\n");
    }

    /// Un nombre non fini s'écrit `null` ; les autres se relisent exactement (`1.0`, `1e-5`).
    #[test]
    fn the_numbers_are_valid_json() {
        assert_eq!(Json::Nombre(f64::NAN).en_texte(), "null\n");
        assert_eq!(Json::Nombre(f64::INFINITY).en_texte(), "null\n");
        assert_eq!(Json::Nombre(1.0).en_texte(), "1.0\n");
        assert_eq!(Json::Nombre(1e-5).en_texte(), "1e-5\n");
        assert_eq!(Json::Entier(-3).en_texte(), "-3\n");
    }
}
