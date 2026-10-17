"""Unit tests for normalize.py on invented examples. Run: python src/test_normalize.py
"""

from normalize import name_core, norm_address, norm_name


def check(got, want):
    assert got == want, f"\n got: {got!r}\nwant: {want!r}"


def test_india():
    check(norm_name("Sunrise Agro Pvt. Ltd."), "sunrise agro private limited")
    check(norm_name("SUNRISE AGRO PRIVATE LÍMITED"), "sunrise agro private limited")
    check(name_core(norm_name("Neel Tráders Pvt. Ltd.")), "neel traders")
    check(norm_name("Mehta & Sons"), "mehta and sons")
    check(norm_name("Kaveri Textiles [Exports]"), "kaveri textiles exports")
    hi = norm_name("आदर्श ट्रेडर्स प्राइवेट लिमिटेड")
    check(hi, "आदर्श ट्रेडर्स प्राइवेट लिमिटेड")
    check(name_core(hi), "आदर्श ट्रेडर्स")
    check(norm_name("రామ ట్రేడర్స్"), "రామ ట్రేడర్స్")
    check(norm_address("45, 2ND FLR, PL-7, SHANTI BLDG ANDHERI, MUMBAI"),
          "45 2nd flr place 7 shanti bldg andheri mumbai")
    check(norm_address("No 12/4 B 2Nd Cross Road Jayanagar"),
          "12 4 b 2nd cross road jayanagar")


def test_us():
    check(norm_name("BX MERIDIAN L.L.C."), "bx meridian llc")
    check(norm_name("Harbor  Point Studio L.L.C."), "harbor point studio llc")
    check(name_core(norm_name("Lakeside Optics LLC")), "lakeside optics")
    check(norm_name("Pine Valley Supply lnc Center"), "pine valley supply lnc center")
    check(norm_name("Acme Corp"), "acme corporation")
    check(norm_name("Jordan K. Ellis, M.D. Co"), "jordan k ellis md company")
    check(norm_address("#1200 Maple Ln, Springfield, Texas"), "1200 maple lane springfield texas")
    check(norm_address("88 Oak Dr, SPRINGFIELD, IL"), "88 oak drive springfield il")
    check(norm_address("17 Elm Rd, # Apartment 5, Riverton"), "17 elm road apartment 5 riverton")


def test_france():
    check(norm_name("ABC Club S.A.R.L."), "abc club sarl")
    check(norm_name("Tennis Amicale  S.A.S.U."), "tennis amicale sasu")
    check(name_core(norm_name("Établissements Durand Développement")), "etablissements durand developpement")
    check(norm_name("Ets Martin SAS"), "etablissements martin sas")
    check(norm_name("Atelier &  Cie"), "atelier and compagnie")
    check(name_core(norm_name("Lyon Çollectif SAS")), "lyon collectif")
    check(norm_name("Voyage Centre (Frânce) SARL"), "voyage centre france sarl")
    check(norm_address("N° 42 AV DES TILLEULS, ARCVILLE SUR MER"),
          "42 avenue des tilleuls arcville sur mer")
    check(norm_address("Nº 7 Rue Des Lilas, Arcville"), "7 rue des lilas arcville")
    check(norm_address("9 R. Pasteur, Arcville, Nord"), "9 rue pasteur arcville nord")
    check(norm_address("3 IMP DES ROSES, ARCVILLE"), "3 impasse des roses arcville")
    check(norm_address("21 Boulevard de la République, Arcville"),
          "21 boulevard de la republique arcville")
    check(norm_name("SARL"), "sarl")
    check(name_core("sarl"), "sarl")


if __name__ == "__main__":
    for f in [test_india, test_us, test_france]:
        f()
    print("normalize tests OK")
