### Yapay Zeka (AI) Nedir?

**Yapay Zeka (AI)**, insan zekasını taklit eden ve _öğrenme, problem çözme, karar verme_ gibi görevleri yerine getirebilen sistemlerdir. AI, veri işleyerek **kalıpları tanır**, **tahminlerde bulunur** ve zamanla **kendini geliştirir**. Teknolojideki hızlı ilerleme sayesinde yapay zeka, _sağlık, otomotiv, finans_ gibi birçok sektörde yaygın olarak kullanılmaktadır. AI, geleceğin en önemli araçlarından biri olarak kabul edilmektedir.

---

### XAI (Açıklanabilir Yapay Zeka) Nedir?

**Açıklanabilir yapay zeka (XAI)**, yapay zeka sistemlerini daha _"insan dostu"_ hale getirmeyi amaçlayan bir yaklaşımdır. Bu sayede yapay zeka sistemlerinin _nasıl kararlar verdiğini_, _hangi bilgileri kullanarak bu sonuçlara ulaştığını_ daha iyi anlayabiliriz. Özellikle _hayati kararlar_ alırken XAI, güvenilirliği artırır ve insanların bu sistemlere daha fazla güven duymasını sağlar. Başka bir deyişle, **XAI**, yapay zekanın _"büyü gibi"_ görünmesini engelleyerek şeffaf ve anlaşılır bir kullanım sunar.

_Gelin bunu bir örnekle açıklayalım:_

Örneğin, bir bankada çalışıyorsunuz ve bu banka, kime kredi verip vermeyeceği konusunda kararlar almak için makine öğrenimi modelleri kullanıyor. Bir gün _Anadolu'nun ücra bir köyünden_ gelen bir çiftçi kredi başvurusu yapıyor, ancak model çiftçiye kredi verilmemesi gerektiğini söylüyor. Doğal olarak çiftçi bu duruma itiraz ederek sizden nedenini sorar. Siz de modelin arkasındaki karmaşık matematiği çiftçiye açıklayamazsınız. İşte böyle durumlarda **XAI** devreye girerek modelin aldığı kararı açıklanabilir hale getirir.

---

### Peki, Bu Araçlar Nasıl Çalışıyor?

Makine öğrenmesi modelleri, veri bilimcilerin birçok sorunu çözmek için kullandığı güçlü araçlardır. Ancak bu modellerin çoğu _kararlarını açıklamak_ için tasarlanmamıştır. Bu yüzden, modellerin verdiği kararları açıklamak için **XAI tekniklerine** ihtiyaç duyulur. XAI araçları arasında en bilinenleri:

- **LIME** (_Local Interpretable Model-Agnostic Explanations_)
- **SHAP** (_SHapley Additive exPlanations_)

Bu konuyla ilgili _Türkçe kaynaklar_ sınırlıdır. Ben de İngilizce kaynaklardan ve bu kütüphanelerin GitHub'daki orijinal dokümantasyonlarından yararlanarak bu konuyu sizlere açıklamaya çalışacağım. İlerleyen zamanlarda, yeni çıkan **XAI araçlarını** da inceleyip sizlerle paylaşmayı hedefliyorum.

---

### Şimdi Bir LIME Kodu Görelim

Aşağıdaki kodda, `LimeTabularExplainer` sınıfını kullanarak bir Lime nesnesi oluşturuyoruz. Bu sınıfta, `LimeTabularExplainer` fonksiyonuna:

- `training_data` parametresi olarak **X_train**
- `feature_names` parametresi olarak **X_train**'in sütun isimlerini veriyoruz.

Daha sonra, `explain_instance` fonksiyonuna:

- `data_row` parametresi olarak `[80000, 800, 0.1]`
- `predict_fn` parametresi olarak da **model.predict_proba** fonksiyonunu veriyoruz.

Son olarak, `exp.show_in_notebook` fonksiyonunu kullanarak **Lime aracının çıktısını** görüntüleyebiliyoruz.

---

```python
import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=['G', 'KP', 'BO'],
    class_names=['Reddedildi', 'Onaylandı'],
    mode='classification'
)
"""
    G = Gelir
    KP = Kredi Puanı
    BO = Borç Oranı
"""
exp = explainer.explain_instance(
    data_row=np.array([80000, 800, 0.1]),
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True, show_all=False)
```

---

![Lime Output 1](./output_img/lime_output_1.png)

### Simdi gelin bu ciktiyi analiz edelim.

Öncelikle, bizim girdiğimiz örnek için (Gelir: 80.000, Kredi Puanı: 800, Borç Oranı: 0.1) modelimiz kredi başvurusunu 0.88 olasılıkla onaylayacağını söylüyor. LIME aracı ise bu kararın nasıl alındığını açıklıyor.
En solda çıktı olasılıklarını verir. %88 olasılıkla kredi başvurusu onaylanacak, %12 olasılıkla reddedilecek.
Ardından, bu çıktı olasılıklarının kolonlara olan etkilerini gösteriyor.
Örneğin; Bu müşteri için Gelir, kredi almasında pozitif etki ediyor, ancak Borç ve Kredi Puanının kredi almasında negatif etkisi var.

## Şimdi bunu test edelim, örneğin, gelir 68.500'den büyük olduğu durumlarda pozitif etki ediyor. O zaman, bu geliri düşürelim, bakalım ne olacak

---

```python
import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=['G', 'KP', 'BO'],
    class_names=['Reddedildi', 'Onaylandı'],
    mode='classification'
)
"""
    G = Gelir
    KP = Kredi Puanı
    BO = Borç Oranı
"""
exp = explainer.explain_instance(
    data_row=np.array([40000, 800, 0.1]),
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True, show_all=False)
```

![Lime Output 2](./output_img/lime_output_2.png)

Gördüğünüz gibi, gelir 68.500'den küçük olduğu durumda kredi başvurusu reddediliyor. Tabii ki, biz örnek olsun diye geliri çok düşürdük, ancak bu örnekle anladığımız şey gelirin kredi başvurusu onayı için önemli bir faktör olduğu.

Şimdi gelin, bizim bir çiftçimiz vardı ya, onun için de Lime aracını kullanalım.
Malum, ona kredi vermemiştik. :(

---

```python
import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=['G', 'KP', 'BO'],
    class_names=['Reddedildi', 'Onaylandı'],
    mode='classification'
)
"""
    G = Gelir
    KP = Kredi Puanı
    BO = Borç Oranı
"""
exp = explainer.explain_instance(
    data_row=np.array([3000, 600, 0.5]),
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True, show_all=False)
```

![Lime Output 3](./output_img/lime_output_3.png)

Görüleceği üzere, bu çiftçi için kredi başvurusu reddedildi, çünkü geliri düşük, kredi puanı düşük ve borç oranı yüksek. Bu durumda, çiftçiye kredi vermediğimiz için haklı çıkmışız. :)
Çiftci sorunu sormuştu ya, biz de cevabını verdik. :) Şimdi çiftçi gitti, gelirini ve kredi puanını yükseltti, borç oranını düşürdü. Bakalım ne olacak?

```python
import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=['G', 'KP', 'BO'],
    class_names=['Reddedildi', 'Onaylandı'],
    mode='classification'
)
"""
    G = Gelir
    KP = Kredi Puanı
    BO = Borç Oranı
"""
exp = explainer.explain_instance(
    data_row=np.array([90000, 900, 0.1]),
    predict_fn=model.predict_proba
)

exp.show_in_notebook(show_table=True, show_all=False)

```

![Lime Output 4](./output_img/lime_output_4.png)
Banka soydu herhalde :) bir anda geliri bu kadar artti. Görüleceği üzere, çiftçi kredi aldı. :)
Ayrıca şunuda unutmamak gerekirki çıktısı her veri örneği için varklı olması gayet normaldir sonuçta kimisinin geliri yüksektir ama borç oarnı yüksektir kimisinin geliri düşüktür ama borç oranı düşüktür. Bu yüzden çıktılar farklı olacaktır.
