🔬 HATA ANALİZİ | Model: convnext_small
   Cihaz: cuda
============================================================

   Tahminler hesaplanıyor...

────────────────────────────────────────────────────────────
 📊 YAŞ GRUBU × CİNSİYET HATA DAĞILIMI
────────────────────────────────────────────────────────────
  [0-4 Yaş                  ] Erkek | n=  44 | MAE:  6.25 | Median:  5.56 | Bias: -0.93↓
  [0-4 Yaş                  ] Kız   | n=  45 | MAE:  4.30 | Median:  3.69 | Bias: +0.43↑
  [4-10 Yaş                 ] Erkek | n= 278 | MAE:  8.09 | Median:  6.81 | Bias: +3.31↑
  [4-10 Yaş                 ] Kız   | n= 417 | MAE:  6.95 | Median:  5.72 | Bias: +2.98↑
  [10-14 Yaş (Puberte)      ] Erkek | n= 570 | MAE:  7.38 | Median:  6.73 | Bias: -5.57↓
  [10-14 Yaş (Puberte)      ] Kız   | n= 325 | MAE:  8.55 | Median:  7.95 | Bias: -6.95↓
  [14-20 Yaş                ] Erkek | n= 152 | MAE:  6.83 | Median:  5.32 | Bias: -3.04↓
  [14-20 Yaş                ] Kız   | n=  61 | MAE:  8.38 | Median:  6.33 | Bias: +0.94↑

────────────────────────────────────────────────────────────
 ⚠️  EN KÖTÜ 20 TAHMİN (Outlier'lar)
────────────────────────────────────────────────────────────
   1. Gerçek:  180.0 → Tahmin:  128.4 | Hata:  51.6 ay | Kız | 14-20 Yaş
   2. Gerçek:  204.0 → Tahmin:  154.6 | Hata:  49.4 ay | Erkek | 14-20 Yaş
   3. Gerçek:  132.0 → Tahmin:   85.1 | Hata:  46.9 ay | Erkek | 10-14 Yaş (Puberte)
   4. Gerçek:  206.0 → Tahmin:  165.1 | Hata:  40.9 ay | Erkek | 14-20 Yaş
   5. Gerçek:  120.0 → Tahmin:  155.3 | Hata:  35.3 ay | Erkek | 4-10 Yaş
   6. Gerçek:  132.0 → Tahmin:   97.2 | Hata:  34.8 ay | Kız | 10-14 Yaş (Puberte)
   7. Gerçek:   94.0 → Tahmin:  128.3 | Hata:  34.3 ay | Erkek | 4-10 Yaş
   8. Gerçek:  162.0 → Tahmin:  128.0 | Hata:  34.0 ay | Kız | 10-14 Yaş (Puberte)
   9. Gerçek:  132.0 → Tahmin:   99.1 | Hata:  32.9 ay | Kız | 10-14 Yaş (Puberte)
  10. Gerçek:  144.0 → Tahmin:  111.9 | Hata:  32.1 ay | Kız | 10-14 Yaş (Puberte)
  11. Gerçek:  180.0 → Tahmin:  211.6 | Hata:  31.6 ay | Erkek | 14-20 Yaş
  12. Gerçek:   50.0 → Tahmin:   81.0 | Hata:  31.0 ay | Kız | 4-10 Yaş
  13. Gerçek:   82.0 → Tahmin:  110.0 | Hata:  28.0 ay | Kız | 4-10 Yaş
  14. Gerçek:  156.0 → Tahmin:  183.9 | Hata:  27.9 ay | Kız | 10-14 Yaş (Puberte)
  15. Gerçek:  132.0 → Tahmin:  104.2 | Hata:  27.8 ay | Kız | 10-14 Yaş (Puberte)
  16. Gerçek:  150.0 → Tahmin:  122.9 | Hata:  27.1 ay | Kız | 10-14 Yaş (Puberte)
  17. Gerçek:  144.0 → Tahmin:  117.0 | Hata:  27.0 ay | Kız | 10-14 Yaş (Puberte)
  18. Gerçek:  168.0 → Tahmin:  194.9 | Hata:  26.9 ay | Erkek | 10-14 Yaş (Puberte)
  19. Gerçek:  150.0 → Tahmin:  123.3 | Hata:  26.7 ay | Erkek | 10-14 Yaş (Puberte)
  20. Gerçek:   94.0 → Tahmin:  120.6 | Hata:  26.6 ay | Kız | 4-10 Yaş

────────────────────────────────────────────────────────────
 📋 GENEL ÖZET
────────────────────────────────────────────────────────────
  Toplam Vaka     : 1892
  Genel MAE       : 7.48 ay
  Genel Median AE : 6.38 ay
  Genel Bias      : -1.96 ay
  ±6 ay içinde    : %46.7
  ±12 ay içinde   : %81.3
  ±24 ay içinde   : %98.4

✅ Detaylı sonuçlar → 'error_analysis_convnext_small.csv'
