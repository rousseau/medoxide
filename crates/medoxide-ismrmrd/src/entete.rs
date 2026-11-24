//! En-tête XML ISMRMRD : les champs utiles à la reconstruction, et tous les paramètres utilisateur.
//!
//! L'en-tête est lu par `roxmltree`, qui construit un arbre **emprunté** au texte (`Document<'a>`) : on en extrait des
//! valeurs possédées (`String`, nombres) avant de rendre la main, pour que [`Entete`] ne dépende plus du texte.

use std::collections::BTreeMap;

/// Ce que la reconstruction lit dans l'en-tête XML ISMRMRD (premier `encoding`).
#[derive(Debug, Clone, PartialEq)]
pub struct Entete {
    /// Taille de la matrice encodée `[lecture, phase 1, phase 2]` (`encodedSpace/matrixSize` x, y, z).
    pub matrice: [usize; 3],
    /// Champ de vue encodé en mm, dans le même ordre (`encodedSpace/fieldOfView_mm`).
    pub champ_de_vue_mm: [f64; 3],
    /// Indice du centre du k-space le long des deux axes de phase (`encodingLimits/.../center`), s'il est donné.
    pub centre: Option<[usize; 2]>,
    /// Fréquence de résonance (porteuse) en Hz (`experimentalConditions/H1resonanceFrequency_Hz`).
    pub porteuse_hz: Option<f64>,
    /// Temps de répétition en ms (`sequenceParameters/TR`, premier), s'il est donné.
    pub tr_ms: Option<f64>,
    /// Tous les paramètres utilisateur (`userParameterString`, `userParameterDouble`, `userParameterLong`), par nom,
    /// valeur en texte telle qu'écrite.
    pub parametres: BTreeMap<String, String>,
}

/// En-tête illisible ou incomplet.
#[derive(Debug, Clone, PartialEq)]
pub struct ErreurEntete(pub String);

impl std::fmt::Display for ErreurEntete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "en-tête ISMRMRD : {}", self.0)
    }
}

impl std::error::Error for ErreurEntete {}

/// Premier descendant de `noeud` dont le nom local est `nom` (l'espace de noms ISMRMRD est ignoré).
fn enfant<'a, 'i>(noeud: roxmltree::Node<'a, 'i>, nom: &str) -> Option<roxmltree::Node<'a, 'i>> {
    noeud.children().find(|n| n.is_element() && n.tag_name().name() == nom)
}

/// Texte du chemin `chemin` (noms d'éléments successifs) sous `noeud`, sans les espaces autour.
fn texte<'a>(noeud: roxmltree::Node<'a, '_>, chemin: &[&str]) -> Option<&'a str> {
    let mut n = noeud;
    for nom in chemin {
        n = enfant(n, nom)?;
    }
    n.text().map(str::trim)
}

fn nombre<T: std::str::FromStr>(noeud: roxmltree::Node<'_, '_>, chemin: &[&str]) -> Result<T, ErreurEntete> {
    let t = texte(noeud, chemin).ok_or_else(|| ErreurEntete(format!("{} absent", chemin.join("/"))))?;
    t.parse()
        .map_err(|_| ErreurEntete(format!("{} : « {t} » n'est pas un nombre", chemin.join("/"))))
}

fn optionnel<T: std::str::FromStr>(noeud: roxmltree::Node<'_, '_>, chemin: &[&str]) -> Option<T> {
    texte(noeud, chemin)?.parse().ok()
}

impl Entete {
    /// Lit l'en-tête XML ISMRMRD `xml`.
    ///
    /// Erreurs : XML mal formé, `encoding/encodedSpace` absent ou non numérique.
    pub fn depuis_xml(xml: &str) -> Result<Self, ErreurEntete> {
        let doc = roxmltree::Document::parse(xml).map_err(|e| ErreurEntete(format!("XML mal formé : {e}")))?;
        let racine = doc.root_element();
        let encodage = enfant(racine, "encoding").ok_or_else(|| ErreurEntete("encoding absent".into()))?;
        let mut matrice = [0usize; 3];
        let mut champ_de_vue_mm = [0f64; 3];
        for (i, axe) in ["x", "y", "z"].iter().enumerate() {
            matrice[i] = nombre(encodage, &["encodedSpace", "matrixSize", axe])?;
            champ_de_vue_mm[i] = nombre(encodage, &["encodedSpace", "fieldOfView_mm", axe])?;
        }
        let centre = (|| {
            Some([
                optionnel(encodage, &["encodingLimits", "kspace_encoding_step_1", "center"])?,
                optionnel(encodage, &["encodingLimits", "kspace_encoding_step_2", "center"])?,
            ])
        })();
        let porteuse_hz = optionnel(racine, &["experimentalConditions", "H1resonanceFrequency_Hz"]);
        let tr_ms = optionnel(racine, &["sequenceParameters", "TR"]);
        let mut parametres = BTreeMap::new();
        if let Some(utilisateur) = enfant(racine, "userParameters") {
            for p in utilisateur.children().filter(|n| n.is_element()) {
                if let (Some(nom), Some(valeur)) = (texte(p, &["name"]), texte(p, &["value"])) {
                    parametres.insert(nom.to_string(), valeur.to_string());
                }
            }
        }
        Ok(Entete {
            matrice,
            champ_de_vue_mm,
            centre,
            porteuse_hz,
            tr_ms,
            parametres,
        })
    }

    /// Paramètre utilisateur `nom` lu comme un nombre, s'il existe et en est un.
    pub fn parametre_nombre(&self, nom: &str) -> Option<f64> {
        self.parametres.get(nom)?.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
 <experimentalConditions><H1resonanceFrequency_Hz>2063802</H1resonanceFrequency_Hz></experimentalConditions>
 <encoding>
  <encodedSpace><matrixSize><x>115</x><y>115</y><z>45</z></matrixSize>
   <fieldOfView_mm><x>230.0</x><y>230.0</y><z>230.0</z></fieldOfView_mm></encodedSpace>
  <reconSpace><matrixSize><x>1</x><y>1</y><z>1</z></matrixSize>
   <fieldOfView_mm><x>1</x><y>1</y><z>1</z></fieldOfView_mm></reconSpace>
  <encodingLimits>
   <kspace_encoding_step_1><minimum>0</minimum><maximum>114</maximum><center>57</center></kspace_encoding_step_1>
   <kspace_encoding_step_2><minimum>0</minimum><maximum>44</maximum><center>22</center></kspace_encoding_step_2>
  </encodingLimits>
 </encoding>
 <sequenceParameters><TR>500.0</TR><TE>14.0</TE></sequenceParameters>
 <userParameters>
  <userParameterDouble><name>dwellTime</name><value>0.04</value></userParameterDouble>
  <userParameterString><name>plane</name><value>yz</value></userParameterString>
 </userParameters>
</ismrmrdHeader>"#;

    #[test]
    fn lit_l_encodage_la_porteuse_et_les_parametres() {
        let e = Entete::depuis_xml(XML).unwrap();
        // encodedSpace, pas reconSpace (qui vient après et vaut 1 ici)
        assert_eq!(e.matrice, [115, 115, 45]);
        assert_eq!(e.champ_de_vue_mm, [230.0, 230.0, 230.0]);
        assert_eq!(e.centre, Some([57, 22]));
        assert_eq!(e.porteuse_hz, Some(2063802.0));
        assert_eq!(e.tr_ms, Some(500.0));
        assert_eq!(e.parametres.get("plane").map(String::as_str), Some("yz"));
        assert_eq!(e.parametre_nombre("dwellTime"), Some(0.04));
        assert_eq!(e.parametre_nombre("plane"), None);
    }

    #[test]
    fn refuse_un_en_tete_sans_matrice() {
        let sans = XML.replace("<x>115</x>", "");
        let erreur = Entete::depuis_xml(&sans).unwrap_err();
        assert!(erreur.0.contains("matrixSize/x"), "{erreur}");
        assert!(Entete::depuis_xml("<pas du xml").is_err());
    }
}
