### Yapay Zeka (AI) Nedir?

**Yapay Zeka (AI)**, insan zekasını taklit eden ve _öğrenme, problem çözme, karar verme_ gibi görevleri yerine getirebilen sistemlerdir. AI, veri işleyerek **kalıpları tanır**, **tahminlerde bulunur** ve zamanla **kendini geliştirir**. Teknolojideki hızlı ilerleme sayesinde yapay zeka, _sağlık, otomotiv, finans_ gibi birçok sektörde yaygın olarak kullanılmaktadır. AI, geleceğin en önemli araçlarından biri olarak kabul edilmektedir.

---

### XAI (Açıklanabilir Yapay Zeka) Nedir?

**Açıklanabilir yapay zeka (XAI)**, yapay zeka sistemlerini daha _"insan dostu"_ hale getirmeyi amaçlayan bir yaklaşımdır. Bu sayede yapay zeka sistemlerinin _nasıl kararlar verdiğini_, _hangi bilgileri kullanarak bu sonuçlara ulaştığını_ daha iyi anlayabiliriz. Özellikle _hayati kararlar_ alırken XAI, güvenilirliği artırır ve insanların bu sistemlere daha fazla güven duymasını sağlar. Başka bir deyişle, **XAI**, yapay zekanın _"büyü gibi"_ görünmesini engelleyerek şeffaf ve anlaşılır bir kullanım sunar.

_Gelin bunu bir örnekle açıklayalım:_

Örneğin, bir bankada çalışıyorsunuz ve bu banka, kime kredi verip vermeyeceği konusunda kararlar almak için makine öğrenimi modelleri kullanıyor. Bir gün _Anadolu'nun ücra bir köyünden_ gelen bir çiftçi kredi başvurusu yapıyor, ancak model çiftçiye kredi verilmemesi gerektiğini söylüyor. Doğal olarak çiftçi bu duruma itiraz ederek sizden nedenini sorar. Siz de modelin arkasındaki karmaşık matematiği çiftçiye açıklayamazsınız. İşte böyle durumlarda **XAI** devreye girerek modelin aldığı kararı açıklanabilir hale getirir.

---

## Peki, Bu Araçlar Nasıl Çalışıyor?

Makine öğrenmesi modelleri, veri bilimcilerin birçok sorunu çözmek için kullandığı güçlü araçlardır. Ancak bu modellerin çoğu _kararlarını açıklamak_ için tasarlanmamıştır. Bu yüzden, modellerin verdiği kararları açıklamak için **XAI tekniklerine** ihtiyaç duyulur. XAI araçları arasında en bilinenleri:

- **LIME** (_Local Interpretable Model-Agnostic Explanations_)
- **SHAP** (_SHapley Additive exPlanations_)

---

## Şimdi Bir LIME Kodu Görelim

Aşağıdaki kodda, LimeTabularExplainer sınıfını kullanarak bir Lime nesnesi oluşturuyoruz. Bu sınıfta, LimeTabularExplainer fonksiyonuna:

- `training_data` parametresi olarak **X_train**
- `feature_names` parametresi olarak **X_train**'in sütun isimlerini veriyoruz.
- `class_names` parametresi olarak **['Onaylanmadı', 'Onaylandı']** veriyoruz.
- `mode` parametresi olarak **'classification'** veriyoruz.

Daha sonra, `explain_instance` fonksiyonuna:

- `data_row` parametresi olarak veri setindeki bir satırı kullanıyoruz.
- `predict_fn` parametresi olarak da **model.predict_proba** fonksiyonunu veriyoruz.

Son olarak, `exp.show_in_notebook` fonksiyonunu kullanarak **Lime aracının çıktısını** görüntüleyebiliyoruz.

```python
import lime
from lime import lime_tabular
import numpy as np

## Approved = 1
data_point_1 = np.array([1, 30.83, 0.000, 1, 1, industry_label['Industrials'],
                         ethnicity_label['White'], 1.25, 1, 1, 1, 0,
                         citizen_label['ByBirth'], 202, 0])

data_point_1_scaled = scaler.transform([data_point_1])

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns.tolist(),
    class_names=['Onaylanmadı', 'Onaylandı'],
    mode='classification'
)

exp_1 = explainer.explain_instance(
    data_row=data_point_1_scaled[0],
    predict_fn=best_rf_model.predict_proba
)

exp_1.show_in_notebook(show_table=True, show_all=False)
```

## Bu Veri Noktası İçin:

- **Gender**: 1-Erkek
- **Age**: 30-30 yaşında
- **Debt**: 8000.0-8000 borcu var
- **Married**: 1-Evli
- **BankCustomer**: 1-Banka Müşterisi
- **Industry**: 'Industry1'-Endüstri 1
- **Ethnicity**: 'White'-Beyaz
- **YearsEmployed**: 5-5 yıl çalışmış
- **PriorDefault**: 0-Önceki kredi ödemesi yok
- **Employed**: 1-Çalışıyor
- **CreditScore**: 750-Kredi Puanı 750
- **DriversLicense**: 1-Ehliyet var
- **Citizen**: 'ByBirth'-Doğumla Vatandaş
- **ZipCode**: '90210'-90210 posta kodu
- **Income**: 80000-80000 geliri var

Veri noktasıyla birlikte model, kredi başvurusunu onaylayıp onaylamayacağına dair bir tahmin yapıyor ve Lime aracı bu tahminin nasıl alındığını açıklıyor.

---

## Güncel Lime Çıktısı

Aşağıdaki çıktı, LIME aracılığıyla elde edilmiştir ve modelin kredi başvurusunu %88 olasılıkla onaylayacağını göstermektedir.

![Lime Output 1](./output_img/lime_output_1.png)

### Simdi gelin bu ciktiyi analiz edelim.

Öncelikle, bizim girdiğimiz örnek için modelimiz kredi başvurusunu 0.88 olasılıkla onaylayacağını söylüyor. LIME aracı ise bu kararın nasıl alındığını açıklıyor.
En solda çıktı olasılıklarını verir. %88 olasılıkla kredi başvurusu onaylanacak, %12 olasılıkla reddedilecek.
Ardından, bu çıktı olasılıklarının kolonlara olan etkilerini gösteriyor.
Örneğin; Bu müşteri için Gelir, kredi almasında pozitif etki ediyor, ancak Borç ve Kredi Puanının kredi almasında negatif etkisi var.

Şimdi gelin, bizim bir çiftçimiz vardı ya, onun için de Lime aracını kullanalım.
Malum, ona kredi vermemiştik. :(

---

![Lime Output 4](./output_img/lime_output_4.png)
Banka soydu herhalde :) bir anda geliri bu kadar artti. Görüleceği üzere, çiftçi kredi aldı. :)
Ayrıca şunuda unutmamak gerekirki çıktısı her veri örneği için varklı olması gayet normaldir sonuçta kimisinin geliri yüksektir ama borç oarnı yüksektir kimisinin geliri düşüktür ama borç oranı düşüktür. Bu yüzden çıktılar farklı olacaktır.
