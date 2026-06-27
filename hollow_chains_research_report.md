# Hollow Chains: Mikro Dil Modellerinde Akıl Yürütme Formunun İçeriği Aşması

### Ölçek ve Bozulma Eksenlerinde Yapısal Sadakat ile Anlamsal Doğruluğun Ayrışması

**Araştırma planı / metodoloji raporu — v1.0**
Yazar: Eren Ata · Bağlam: SLM güvenilirliği / edge AI · İlişkili altyapı: `cosmic-slm-flightkit`

---

## 1. Özet

Mikro ölçekli (1M–350M) "düşünen" dil modelleri, akıl yürütme izlerini (`<think>` bloklarını) biçimsel olarak kusursuz üretebilirken içerikleri olgusal/mantıksal olarak çökmüş olabilir — gözlemlenen örneklerde model "AI ilk kez 1965'te MIT'de tanıtıldı" gibi akıcı ama tamamen yanlış çözümler verir. Bu çalışma, bir akıl yürütme izindeki **yapısal form (Structural Fidelity, SF)** ile **anlamsal içeriğin (Semantic Correctness, SC)** ayrı, ayrıştırılabilir iki yetenek olduğunu ve bunların **kırılganlık profillerinin zıt** olduğunu öne sürer:

- **Ortaya çıkış ekseninde** (parametre/veri/öğretmen) form *önce* doğar; içerik çok geç gelir ya da hiç gelmez.
- **Yıkım ekseninde** (ağırlık bit-flip + kuantizasyon) form *son* ölür; içerik sessizce çürür.

Birleşik iddia tek bir simetri yasasıdır: **"Form, ortaya çıkışta ilk, yıkımda son."** Yapısal iskele düşük entropili, yüksek-önsel (high-prior) bir alt sistem olduğundan hem öğrenmesi ucuz hem yok etmesi zordur. Sonuç: iyi biçimlenmiş bir `<think>` bloğu, tam da en çok güvenildiği rejimde (küçük + kuantize + soft-error'a açık edge cihazı) sistematik olarak **yanıltıcı bir güven sinyalidir**.

---

## 2. Motivasyon ve problem

Akıl yürütme modellerinin yaygınlaşmasıyla `<think>` izleri, kullanıcı ve sistemler için örtük bir güven sinyaline dönüştü: "model adım adım düşünüyor, demek ki cevabı güvenilir." Bu varsayım, izin **biçimi** ile **doğruluğu** arasında bir bağ olduğunu kabul eder. Mikro modellerde ve bozulmuş modellerde bu bağ kopar: form ayakta kalır, içerik boşalır. Bu "boş kabuk" (hollow chain) tehlikelidir çünkü yapısal sadakat, içeriğin hak etmediği bir güveni taşır.

Problem iki cephede aynıdır:
- **Edge dağıtımı:** Küçük + kuantize modeller tam da kaynak-kısıtlı, soft-error'a açık donanımda çalışır — formun içerikten kopması en kritik olduğu yer burasıdır.
- **Değerlendirme:** Mevcut otomatik metrikler (cevap doğruluğu *veya* yüzeysel akıcılık) iki ekseni birbirine karıştırır; "iyi biçimli ama yanlış" örnekler raporlarda görünmez kalır.

---

## 3. Kavramsal çerçeve

Bir akıl yürütme izi iki ortogonal eksende ölçülür:

- **Structural Fidelity (SF) ∈ [0,1]:** İzin biçimsel doğruluğu — etiketlerin varlığı/sırası/kapanışı, bölümleme oranı, uzunluk dağılımına uyum, şablon n-gram kalıplarına benzerlik, token-entropisi profili, tekrar/dejenerasyon. *İçerikten bağımsız* hesaplanır.
- **Semantic Correctness (SC) ∈ [0,1]:** İçeriğin doğruluğu — nihai cevap doğruluğu, adım-düzeyi geçerlilik, olgusal atomik doğrulama, içsel çelişki yokluğu. *Formdan bağımsız* hesaplanır.

Türetilen büyüklükler:

- **Form–Substance Gap:** `FSG = SF − SC` (normalize). Pozitif ve büyük FSG = tiyatro.
- **Theater Score:** `SF ≥ τ_high ∧ SC ≤ τ_low` koşulunu sağlayan örneklerin oranı — çalışmanın başlık metriği.
- **Dört-yönlü sınıflama:** her örnek {coherent-correct, **coherent-wrong (theater)**, malformed-correct, malformed-wrong} kümesine atanır.

Bu üç büyüklük **hem ölçek hem bozulma eğrilerinde aynı eksende** okunur; iki yarımı tek anlatıda birleştiren tutkal budur.

---

## 4. Birleşik tez ve simetri hipotezi

İki deney, ayrı çalışmalar değil, tek olgunun (SF/SC ayrışması) iki probudur:

| | Ortaya çıkış ekseni (Faz A) | Yıkım ekseni (Faz B) |
|---|---|---|
| Değişken | parametre / veri / öğretmen boyutu | bit-flip oranı / kuantizasyon seviyesi |
| Soru | İçerik *olmadan* form ne zaman doğar? | Form *kaybolmadan* içerik ne zaman ölür? |
| Beklenti | SF erken doygunlaşır, SC geç tırmanır | SC erken çöker, SF geç çöker |

**Simetri yasası (ana hipotez):** Aynı düşük-entropi/yüksek-önsel mekanizma, formu hem *erken öğrenilen* hem *geç bozulan* yapar. Dolayısıyla bir modelin ortaya çıkıştaki "form-içerik açığı" ile yıkımdaki "form dayanıklılık marjı" pozitif ilişkilidir.

---

## 5. Ölçüm katmanı (operasyonel tanımlar)

Tüm metrikler, standart bir **üretim kaydı şeması** (JSONL) üzerinde çalışır:

```json
{"id": "...", "prompt": "...", "task_type": "arithmetic|symbolic|factual_mcq",
 "gold": "...", "generation": "<|begin_of_thought|>...", 
 "think": "...", "solution": "...", "token_entropies": [..], "meta": {...}}
```

### 5.1 Structural Fidelity (SF) — alt metrikler
1. **parse_rate** — gerekli etiketlerin doğru sırada ve kapalı olma oranı (durum makinesi/regex; örnek başına ikili → ortalama).
2. **tag_validity** — `<|begin_of_thought|>…<|end_of_thought|><|begin_of_solution|>…<|end_of_solution|>` bileşen kontrolü: varlık, sıra, tek-oluşum, iç içe geçme hatası yok.
3. **section_ratio_conformity** — `think_len / (think_len + solution_len)` örnek dağılımının referans (öğretmen) dağılıma uzaklığı (Wasserstein).
4. **length_conformity** — üretim uzunluk dağılımı ↔ referans dağılım Wasserstein mesafesi.
5. **template_ngram_overlap** — açılış n-gramlarının öğretmen şablonlarıyla örtüşmesi ("Okay, the user wants…", "Let me start by recalling…", "So the key points are…").
6. **entropy_profile** — think-bloğu vs. solution-bloğu ortalama token entropisi (form token'ları düşük entropi eğilimli).
7. **repetition** — distinct-n / rep-n dejenerasyon ölçütleri.
8. **SF (kompozit)** — yukarıdakilerin normalize ve dokümante edilmiş ağırlıklı birleşimi; **bileşenlerle birlikte** raporlanır (tek sayıya gömülmez).

### 5.2 Semantic Correctness (SC) — alt metrikler
1. **answer_accuracy** — solution bloğundan çıkarılan nihai cevabın altın değere eşitliği (math: exact/normalized; MCQ: etiket eşleşmesi).
2. **step_validity** — aritmetik: denklemleri ayrıştırıp her adımı doğrula; sembolik: kural-tabanlı kontrol. Geçerli adım oranı.
3. **factual_atomic** — olgusal görevlerde iddiayı atomik parçalara ayırıp kürate edilmiş altın-olgu kümesine karşı doğrula (ilk milestone'da *kapalı-uçlu, deterministik doğrulanabilir* görevlerle sınırlı tut).
4. **contradiction** — think ↔ solution NLI çelişkisi (opsiyonel; model gerektirir, GPU/Colab).
5. **SC (kompozit)** — normalize birleşim; bileşenlerle birlikte raporlanır.

### 5.3 Açık (gap) metrikleri
- **FSG = SF − SC**, **theater_score**, **dört-yönlü sınıflama**, ve örnek-başına ortak (SF, SC) saçılımı.

> **Tasarım ilkesi:** İlk milestone'da SC'yi *deterministik doğrulanabilir* görevlere (aritmetik, sembolik, çoktan-seçmeli olgu) sabitle. Böylece SC ölçümü model-yargıcına bağlı kalmaz ve metrik katmanı tamamen CPU'da, GPU'suz birim testleriyle doğrulanabilir.

---

## 6. Deneysel tasarım

### Faz 0 — Ölçüm katmanı ve veri şeması *(yerel, CPU)*
SF/SC/gap kütüphanesi + birim testleri + JSONL şeması + görev yükleyiciler. GPU'suz tamamen test edilebilir. **İlk Cursor milestone'u budur.**

### Faz A — Ortaya çıkış taraması *(Colab GPU)*
- **Model merdiveni:** Tek mimari ailede ~1M → 8M → 50M → 150M → 350M (Llama-benzeri, tam kontrol için kendi eğittiğin). SupraLabs/TinyStories aileleri *replikasyon/dış geçerlilik* için ikincil eksen.
- **Reasoning indükleme:** Base (FineWeb-Edu altkümesi) → SFT(`<think>` formatı). Kontrollü değişkenler:
  - **Öğretmen boyutu** (0.5B / 1.7B / 7B): büyük öğretmen SC'yi mi yoksa sadece SF'yi mi taşır?
  - **Örnek sayısı** (50 / 500 / 5k / 50k): *formu öğretmek için kaç örnek, içeriği öğretmek için kaç örnek (ya da hiç)?*
  - **Epoch** (2 / 6 / 20): aşırı SFT formu pekiştirip içeriği ezberletir mi?
  - **Format-kısıtı kolu:** equation-only (literatürde T5-Tiny'de etkili) — form basitleşince FSG kapanıyor mu?
- **Çıktı:** SF(scale) ve SC(scale) eğrileri; aradaki taralı alan = "tiyatro bölgesi".

### Faz B — Yıkım taraması *(Colab GPU; enjeksiyon mantığı yerel)*
- **Bozulma türleri:** (a) ağırlık bit-flip (katman/tensör/bit-pozisyonu hedefli; `cosmic-slm-flightkit` enjektörünü *yeniden kullan/genişlet*), (b) kuantizasyon (GGUF Q8→Q4→Q2; SupraLabs zaten GGUF yayınlıyor), (c) BitNet/1-bit kolu.
- **Ayrıştırma:** bit-flip'i tensör tipine (embedding / attention / MLP / çıkış-başı) ve bit-pozisyonuna (exponent üst bit vs. mantissa) göre ayır.
- **Çıktı:** Her bozulma seviyesinde SF(p), SC(p). Kritik nicelik: **SC, SF'den önce mi çöküyor?** "Sessiz bölge" genişliği = SDC'nin aldatıcılık marjı.
- **Yeni kavram:** *structural-vs-semantic SDC* — bozulmayı tek perplexity sayısına indirmek yerine iki kanala ayırma.

### Faz C — Birleştirme *(yerel analiz)*
- 2B ısı-haritası: x = yetenek gradyanı (A), z = bozulma gradyanı (B), renk = theater_score.
- **Simetri testi:** A'daki form-SC açığı ile B'deki form dayanıklılık marjı arasındaki korelasyon.
- **Mekanik destek (opsiyonel):** form-token logit/entropi dağılımının içerik-token'larından sistematik farkı (simetriyi *açıklayan* mekanizma).
- **Pratik çıkarım:** sadece SF'ye bakan güven sinyallerinin edge'de tehlikesini gösteren hafif "theater detector".

### Faz D — Hipotez testleri
Aşağıdaki H1–H4 üzerinde istatistiksel testler ve güven aralıkları.

---

## 7. Doğrulanabilir hipotezler

- **H1 (erken form):** SF, SC'nin %50'sine ulaşmasından ≥1 büyüklük mertebesi daha az parametrede platoya ulaşır.
- **H2 (geç ölüm):** Bozulma altında SC, SF'den daha düşük bir bozulma oranında kritik eşiğin altına düşer.
- **H3 (simetri):** H1'deki açık ile H2'deki dayanıklılık marjı pozitif korelasyon gösterir.
- **H4 (müdahale):** equation-only / form-basitleştirme hem ortaya çıkışta FSG'yi daraltır hem yıkımda anlamsal dayanıklılığı artırır.

---

## 8. Literatür konumlanması

- **CoT faithfulness'tan farkı:** Faithfulness, izin *iç hesabı* yansıtıp yansıtmadığını sorar (Turpin, Lanham); ilginç biçimde bozuk akıl yürütmeyle bile cevap çoğu zaman doğru çıkabilir. Biz iz-*formu* ile içerik-*doğruluğu* arasındaki **davranışsal, dışarıdan ölçülebilir** ilişkiyi sorarız; iç mekanizma iddiası gerektirmez. Ayrıca faithfulness çalışmaları *metinsel* karşı-olgu kullanır; biz *ağırlık-düzeyi donanım bozulması* kullanırız (farklı tehdit modeli).
- **TinyStories soyundan farkı:** O çizgi mikro modellerde *akıcılık/tutarlılık* ölçer; biz *olgusal/mantıksal doğruluğu formdan ayırarak* ölçeriz — sub-100M rejiminde olgusallık literatürünün dokunmadığı yer.
- **CoT-distillation'dan farkı:** O çizgi küçük modelde reasoning'i *iyileştirmeyi* hedefler; biz formun içerik olmadan öğrenilebildiğini *karakterize* ederiz.
- **SDC/bit-flip'ten farkı:** Mevcut SDC işi agregat doğruluk/perplexity düşüşü raporlar; biz bozulmayı yapısal vs. anlamsal kanala ayrıştırırız.

---

## 9. Katkılar ve çıktılar

1. **İki-eksenli SF/SC çerçevesi** + açık ölçüm kiti (CPU'da koşan, hem formu hem içeriği raporlayan, NanoBEIR ruhunda hafif).
2. **Ampirik 2B harita** — tiyatro bölgesinin ölçek ve bozulma boyunca konumu.
3. **Simetri yasası** ("form: ortaya çıkışta ilk, yıkımda son") + onu açıklayan düşük-entropi mekanizması.
4. **"structural-vs-semantic SDC"** kavramı ve edge için theater-detector önerisi.

---

## 10. Altyapı ve iş bölümü

| Katman | Nerede | Ne |
|---|---|---|
| Kütüphane kodu (`src/`) | Yerel (Cursor) | Model ladder, metrikler, bozulma, analiz — *import edilebilir* paket |
| Metrik/analiz/çizim | Yerel `.py`, CPU | SF/SC/gap, agregasyon, 2B harita, figürler |
| Bit-flip enjeksiyon mantığı | Yerel `.py` | Ağırlık manipülasyonu (CPU tensör ops); flightkit ile hizalı |
| Eğitim (pretrain + SFT) | Colab `.ipynb`, GPU | Model merdiveni, öğretmen/örnek/epoch süpürmeleri |
| Üretim (generation) | Colab `.ipynb`, GPU | Değerlendirme için forward pass; bozuk-model üretimi |

**Prensip:** Notebook'lar *ince* kalır — repoyu (`pip install -e .`) import edip orkestrasyon yapar; tüm mantık `src/`'tedir. Böylece aynı kod hem yerelde test edilir hem Colab'da koşar.

---

## 11. Önerilen depo yapısı

```
hollow-chains/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model_ladder.yaml      # 1M..350M arch hedefleri
│   ├── sft.yaml               # öğretmen / örnek-sayısı / epoch / format kolları
│   └── corruption.yaml        # bit-flip oranları, quant seviyeleri, hedef katmanlar
├── src/hollow_chains/
│   ├── data/{build_reasoning_sft.py, tasks.py, schema.py}
│   ├── models/{ladder.py, registry.py}
│   ├── train/{pretrain.py, sft.py}
│   ├── metrics/{structural.py, semantic.py, gap.py}
│   ├── corruption/{bitflip.py, quantize.py}
│   ├── eval/{run_emergence.py, run_degradation.py, aggregate.py}
│   └── viz/plots.py
├── scripts/{prepare_data.py, compute_metrics.py, make_figures.py}   # ince yerel giriş noktaları
├── notebooks/{00_setup_colab.ipynb, 01_pretrain_ladder.ipynb,
│              02_sft_sweeps.ipynb, 03_corruption_sweeps.ipynb}
└── tests/test_metrics.py
```

---

## 12. Yol haritası (milestone'lar)

- **M1 — Ölçüm katmanı (Faz 0):** repo iskeleti + SF/SC/gap + şema + görev yükleyiciler + birim testleri. *Yerel, GPU'suz, hemen test edilebilir.* ← ilk Cursor promptu.
- **M2 — Model merdiveni + eğitim (Faz A):** ladder.py + pretrain/sft harness + Colab pretrain & SFT notebook'ları.
- **M3 — Bozulma (Faz B):** bitflip/quantize + degradation notebook'ları (flightkit ile entegrasyon).
- **M4 — Birleştirme (Faz C/D):** aggregate + 2B harita + simetri testi + figürler.

---

## 13. Riskler ve önlemler

- **SC gürültüsü:** kapalı-uçlu, otomatik doğrulanabilir görevlere ağırlık ver; LLM-as-judge ikincil + insan-doğrulamalı.
- **Mimari karışması:** tek aile, sabit tokenizer/bağlam; SupraLabs/TinyStories sadece replikasyon.
- **Bit-flip varyansı:** çok-tohumlu enjeksiyon, güven aralıkları.
- **"Negatif sonuç" algısı:** çerçeveyi tanısal araç + güvenlik uyarısı olarak konumla; tiyatronun *yokluğu* da bulgudur.
- **Vocab baskınlığı (mikro ölçek):** 1M civarında embedding parametreyi domine eder → küçük vocab (~8–16k, kendi BPE'n veya kırpılmış) kullan.

---

## 14. Dayanaklar (anahtar referanslar)

- TinyStories (Eldan & Li, 2023) — sub-10M modellerde akıcı/tutarlı üretim; yetenek hiyerarşisi (dilbilgisi→tutarlılık→akıl yürütme).
- CoT (Wei et al., 2022) — küçük ölçekte "akıcı ama mantıksız" düşünce zincirleri.
- Teaching SLMs to Reason (Magister et al., 2023) — <10B'de CoT doğruluğu düşürebilir; damıtma.
- Small LMs are Equation Reasoners (2024) — equation-only format çok küçük modellerde aritmetiği iyileştirir.
- CoT Faithfulness (Turpin 2023; Lanham 2023) — bozuk akıl yürütmeyle bile doğru cevap; *form-içerik ayrışmasından farklı eksen*.
- SDC / bit-flip & 1-bit (BitNet, TernaryLM) — kuantize/edge rejiminde dayanıklılık; mevcut işin agregat ölçtüğü yer.

*(Tam künyeler yazım aşamasında BibTeX'e dökülecek.)*
