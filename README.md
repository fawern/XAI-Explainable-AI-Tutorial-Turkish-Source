### Yapay Zeka ve XAI (Explainable Artificial Intelligence) Nedir?

- Yapay zeka, bilgisayarların insanların yaptığı gibi düşünmesini sağlayan bir bilim dalıdır. Bildiğiniz üzere, Chat-GPT gibi araçların da çıkmasıyla yapay zeka hayatımıza iyice girmeye başladı. Biz farkında olsak da olmasak da her yerde bu araçları kullanıyoruz. Peki, siz hayatımızın her yerinde kullandığımız bu araçların nasıl çalıştığını merak ettiniz mi?

---

### XAI (Explainable Artificial Intelligence) Nedir?

- Açıklanabilir yapay zeka (XAI), yapay zeka sistemlerini daha "insan dostu" hale getirmeyi amaçlayan bir yaklaşımdır. Yapay zeka sistemlerinin nasıl kararlar verdiğini, hangi bilgileri kullanarak bu kararları aldığını ve neden belirli sonuçlara ulaştığını daha iyi anlayabiliriz. Bunu yapmak, özellikle hayati öneme sahip kararlar alındığında, güvenilirliği artırır ve insanların bu sistemlere daha fazla güven duymasına yardımcı olur. Başka bir deyişle, XAI, yapay zekanın "büyü gibi" görünmesini önler ve insanların bu teknolojiyi daha şeffaf ve anlaşılır bir şekilde kullanmasına yardımcı olur.

Basit bir örnek vermek gerekirse;

- Günümüzde bazı şirketler artık yüzlerce iş başvurusu için tek tek CV'leri incelemek yerine yapay zeka araçlarıyla CV'leri analiz ediyorlar ve sizin bu iş pozisyonuna uygun olup olmadığınızı belirliyorlar. Ancak, bu araçların çoğu kararlarını açıklamak için tasarlanmamıştır. Bu nedenle, bu araçların kararlarını açıklamak için Açıklanabilir Yapay Zeka (XAI) tekniklerine ihtiyaç duyulmaktadır. Neden mi? Çünkü düşünün ki bir şirkete başvuru yaptınız, CV'niz incelendi ve bu iş için uygun olmadığınız belirlendi ancak neden uygun olmadığınızı bilmiyorsunuz ve size söylenmedi. Bu durumda size bir haksızlık yapıldığını düşünmez misiniz? Ya da şirket size doğrudan "Bizim yapay zeka modelimiz sizi uygun bulmadı" derse, bu oldukça absürt olmaz mı? Siz de demez misiniz, neden beni uygun bulmadı diye?

Bir örnek daha vermek gerekirse;

- Örneğin, siz bir bankada çalışıyorsunuz ve bu banka, kime kredi verip vermeme konusunda kararlar almak için basit makine öğrenimi araçlarını kullanıyor. Bir gün Anadolu'dan gelmiş bir çiftçi kredi başvurusu yapıyor ve bu araç çiftciye kredi verilmemesi gerektiğini söylüyor. Doğal olarak çiftçi, sizden bunun nedenini sorar. Malum, çiftçinin ekini var, ülkenin durumu belli :) nasıl açıklarsınız bu durumu oturup çiftçiye bu modelin arkasındaki matematiği anlatacak değilsiniz herhalde :) Adam demez mi, "Ya toprağımı başlatma, şimdi cebirinden ve kalkülüsünden bana nedenini söyle." Siz bu durumu çiftçiye nasıl açıklarsınız?

İşte bu gibi durumlardada XAI teknikleri devreye giriyor ve sizin neden uygun olmadığınızı açıklıyor.

---

### Peki, Bu Araçların Nasıl Çalıştığını Merak Ettiniz mi?

- Makine öğrenmesi modelleri, veri bilimcilerin ve mühendislerin birçok sorunu çözümünde kullandığı bir araçtır. Ancak bu modellerin çoğu, kararlarını açıklamak için tasarlanmamıştır. Bu nedenle, bu modellerin kararlarını açıklamak için XAI tekniklerine ihtiyaç duyulmaktadır.
- Bu araçların başında ise LIME (Local Interpretable Model-Agnostic Explanations) ve SHAP (SHapley Additive exPlanations) gelmektedir.
- Bu alanda Türkçe kaynak neredeyse yok, ben sadece İngilizce kaynaklardan ve bu kütüphanelerin GitHub'daki orijinal dokümantasyonlarından yararlanarak sizlere bu konuyu anlatmaya çalışacağım.
- İleriki zamanlarda bu iki araç dışında yeni çıkan araçları da inceleyip sizlerle paylaşacağım.

---

### Simdi bir Lime kodu gorelim

- Asagidaki kodda, LimeTabularExplainer sınıfını kullanarak bir Lime nesnesi oluşturuyoruz.
- Bu sınıfın içerisindeki LimeTabularExplainer fonksiyonuna training_data parametresi olarak X_train'i, feature_names parametresi olarak da X_train'in sütun isimlerini veriyoruz.
- Daha sonra, LimeTabularExplainer sınıfının içerisindeki explain_instance fonksiyonuna data_row parametresi olarak [80000, 800, 0.1] değerlerini, predict_fn parametresi olarak da model.predict_proba fonksiyonunu veriyoruz.
- Daha sonra, exp.show_in_notebook fonksiyonunu kullanarak Lime aracının çıktısını görebiliyoruz.

---

```python
import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
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

Öncelikle, bizim girdiğimiz örnek için (Gelir: 80.000, Kredi Puanı: 800, Borç Oranı: 0.1) modelimiz kredi başvurusunu 0.9999 olasılıkla onaylayacağını söylüyor. LIME aracı ise bu kararın nasıl alındığını açıklıyor.
En solda çıktı olasılıklarını verir. %88 olasılıkla kredi başvurusu onaylanacak, %12 olasılıkla reddedilecek.
Ardından, bu çıktı olasılıklarının kolonlara olan etkilerini gösterir.
Örneğin; Bu müşteri için Gelir, kredi almasında pozitif etki ediyor, ancak Borç ve Kredi Puanının kredi almasında negatif etkisi var.

## Şimdi bunu test edelim, örneğin, gelir 68.500'den büyük olduğu durumlarda pozitif etki ediyor. O zaman, bu geliri düşürelim, bakalım ne olacak

- Çıktımız 0.7 ile onaylandı sınıfını tahmin etmiş. Bu tahminin nedenini ise aşağıdaki şekilde açıklıyor:
  Müşterinin borç oranı 0.35 (sağdaki tabloda 0.35 değerini görebilirsiniz) ve bu değer 0.38 değerinden küçük ve 0.25 değerinden büyük olduğu için onaylandı sınıfını tahmin etmiş.
  Müşterinin KP (kredi puanı) değeri 740 ve bu değer 745 değerinden küçük ve 720 değerinden büyük olduğu için onaylandı sınıfını tahmin etmiş.
  Müşterinin G (gelir) değeri 650.000 ve bu değer 685.000 değerinden küçük ve 59.000 değerinden büyük olduğu için onaylandı sınıfını tahmin etmiş.

---

```python

import lime
from lime import lime_tabular

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
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

Gördüğünüz gibi, gelir 68.500'den küçük olduğu durumda kredi başvurusu reddediliyor. Tabii ki, biz örnek olsun diye geliri çok düşürdük, ancak bu örnekle anladığımız şey gelirin kredi başvurusu için çok önemli olduğu.
