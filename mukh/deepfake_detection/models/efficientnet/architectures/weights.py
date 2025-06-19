"""
Video Face Manipulation Detection Through Ensemble of CNNs

Author Credits:

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini

GitHub: https://github.com/polimi-ispl/icpr2020dfdc
License: GNU GENERAL PUBLIC LICENSE 3.0
"""

weight_url = {
    "EfficientNetAutoAttB4ST_DFDC": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4ST_DFDC_bestval-4df0ef7d2f380a5955affa78c35d0942ac1cd65229510353b252737775515a33.pth",
    "EfficientNetAutoAttB4ST_FFPP": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4ST_FFPP_bestval-ddb357503b9b902e1b925c2550415604c4252b9b9ecafeb7369dc58cc16e9edd.pth",
    "EfficientNetAutoAttB4_DFDC": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth",
    "EfficientNetAutoAttB4_FFPP": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4_FFPP_bestval-b0c9e9522a7143cf119843e910234be5e30f77dc527b1b427cdffa5ce3bdbc25.pth",
    "EfficientNetB4ST_DFDC": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4ST_DFDC_bestval-86f0a0701b18694dfb5e7837bd09fa8e48a5146c193227edccf59f1b038181c6.pth",
    "EfficientNetB4ST_FFPP": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4ST_FFPP_bestval-ccd016668071be5bf5fff68e446d055441739ec7113fb1a6eee998f08396ae92.pth",
    "EfficientNetB4_DFDC": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4_DFDC_bestval-c9f3663e2116d3356d056a0ce6453e0fc412a8df68ebd0902f07104d9129a09a.pth",
    "EfficientNetB4_FFPP": "https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4_FFPP_bestval-93aaad84946829e793d1a67ed7e0309b535e2f2395acb4f8d16b92c0616ba8d7.pth",
    "Xception_DFDC": "https://f002.backblazeb2.com/file/icpr2020/Xception_DFDC_bestval-e826cdb64d73ef491e6b8ff8fce0e1e1b7fc1d8e2715bc51a56280fff17596f9.pth",
    "Xception_FFPP": "https://f002.backblazeb2.com/file/icpr2020/Xception_FFPP_bestval-bb119e4913cb8f816cd28a03f81f4c603d6351bf8e3f8e3eb99eebc923aecd22.pth",
}
