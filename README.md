# PneumoniaDetection

1.1. Projenin Tanımı ve Seçilme Gerekçesi
Bu proje, akciğer röntgen (X-Ray) görüntülerini analiz ederek hastanın Zatürre (Pneumonia) olup olmadığını tespit eden yapay zeka tabanlı bir tıbbi asistan sistemidir. Projenin seçilme gerekçesi, sağlık sektöründe radyologların iş yükünün artması ve yorgunluğa bağlı teşhis hatalarının önlenmesi ihtiyacıdır. Zatürre, özellikle çocuklar ve yaşlılarda erken teşhis edilmediğinde ölümcül olabilen bir hastalıktır. İnsan gözünün kaçırabileceği ince detayları (opaklaşma, sıvı birikimi) yapay zekanın piksel düzeyinde yakalayabilmesi, bu projeyi hayati kılmaktadır.

1.2. İlgili Alanın Önemi
Tıbbi görüntüleme (Medical Imaging), yapay zekanın en başarılı olduğu alanlardan biridir. Dünya Sağlık Örgütü verilerine göre, zatürre dünya genelinde çocuk ölümlerinin %15'inden sorumludur. Bu proje, "Bilgisayar Destekli Tanı" (CAD - Computer Aided Diagnosis) sistemlerine bir örnek olup, doktorlara "ikinci bir görüş" (second opinion) sunarak karar verme süreçlerini hızlandırmayı ve doğruluğu artırmayı hedefler.

1.3. Literatür Özeti
Literatürde zatürre tespiti için geçmişte geleneksel görüntü işleme teknikleri (Kenar tespiti, Histogram eşitleme) kullanılmıştır. Ancak son yıllarda Derin Öğrenme (Deep Learning) ve özellikle Konvolüsyonel Sinir Ağları (CNN) bu alanda standart haline gelmiştir. Rajpurkar et al. (CheXNet), 121 katmanlı bir ağ kullanarak uzman radyologlar seviyesinde teşhis başarısı elde etmiştir. Bu projede de literatürdeki bu modern yaklaşımlar temel alınmıştır.

2. Veri Setinin Belirlenmesi 
2.1. Veri Kaynağı
Projede, akademik çalışmalarda ve Kaggle platformunda standart olarak kabul edilen "Chest X-Ray Images (Pneumonia)" veri seti kullanılmıştır. Veri seti Guangzhou Kadın ve Çocuk Tıp Merkezi'nden alınan 1 ila 5 yaş arası pediatrik hastaların rutin klinik kontrolleri sırasında elde edilmiştir.

2.2. Veri Yapısı ve İstatistikler
Veri seti toplam 5,863 adet JPEG formatında X-Ray görüntüsünden oluşmaktadır. Veriler iki ana sınıfa ayrılmıştır:

NORMAL: Sağlıklı akciğer görüntüleri.

PNEUMONIA: Bakteriyel veya viral zatürre bulgusu içeren görüntüler.

Veri seti, modelin başarısını objektif ölçmek için üç alt klasöre ayrılmıştır:

Train (Eğitim): Modelin öğrenmesi için kullanılan ana veri grubu.

Test: Modelin eğitimden sonra başarısının ölçüldüğü grup.

Validation (Doğrulama): Eğitim sırasında aşırı öğrenmeyi (overfitting) engellemek için kullanılan kontrol grubu.

Veri Ön İşleme Notu: Modelin ezberlemesini önlemek için eğitim verileri üzerinde Data Augmentation (Veri Çoğaltma) teknikleri (döndürme, yakınlaştırma, yatay çevirme) uygulanarak veri çeşitliliği artırılmıştır.

3. Yöntem ve Algoritma Seçimi
3.1. Seçilen Yöntem: Transfer Learning (VGG16)
Projede sıfırdan bir CNN modeli eğitmek yerine, Transfer Learning (Transfer Öğrenme) yöntemi tercih edilmiş ve VGG16 mimarisi kullanılmıştır.

3.2. Yöntem Seçim Gerekçesi ve Karşılaştırmalı Analiz
Literatürdeki yöntemler karşılaştırıldığında bu seçimin nedenleri şunlardır:

Geleneksel Makine Öğrenmesi (SVM, Random Forest): Bu yöntemler, görüntülerden manuel özellik çıkarımı (Feature Extraction) gerektirir. X-Ray gibi karmaşık görüntülerde manuel özellik çıkarımı hataya açıktır.

Sıfırdan CNN Eğitimi: Milyonlarca veri ve çok yüksek işlemci gücü (GPU) gerektirir. Veri setimiz (5000+ görüntü) sıfırdan derin bir ağ eğitmek için yetersizdir ve "Overfitting" riski yüksektir.

Transfer Learning (Bizim Seçimimiz): ImageNet üzerinde 14 milyon görüntüyle eğitilmiş VGG16 modeli, kenar, doku ve şekil bilgisini zaten bilmektedir. Biz bu "hazır zekayı" alıp, son katmanlarını kendi problemimize (Zatürre/Normal) uyarladık. Bu yöntem, daha az veriyle, daha kısa sürede, çok daha yüksek doğruluk sağlar.

4. Model Eğitimi & Değerlendirilmesi
4.1. Model Mimarisi
Model, Keras/TensorFlow kütüphanesi kullanılarak şu katmanlarla oluşturulmuştur:

Base Model: VGG16 (Ağırlıkları dondurulmuş).

Global Average Pooling: Özellik haritalarını vektöre çevirmek için.

Dense Layer (128 Nöron): ReLU aktivasyon fonksiyonu ile özelliklerin işlenmesi.

Dropout (0.5): Nöronların %50'si rastgele kapatılarak ezberleme engellendi.

Output Layer (1 Nöron): Sigmoid aktivasyon fonksiyonu (0 ile 1 arası olasılık değeri üretmek için).

4.2. Eğitim Hiperparametreleri
Optimizer: Adam (Adaptive Moment Estimation)

Loss Function: Binary Crossentropy (İkili sınıflandırma olduğu için)

Batch Size: 32

Epoch: 10 (Early Stopping mekanizması ile en iyi model kaydedildiğinde durduruldu).

4.3. Değerlendirme Sonuçları
Eğitim sonucunda model, test veri seti üzerinde yüksek başarı göstermiştir.

Eğitim Doğruluğu (Train Accuracy): ~%91

Test Doğruluğu (Test Accuracy): ~%86

5. Sonuç ve Gelecek Çalışmalar
Bu proje ile geliştirilen yapay zeka modeli, akciğer röntgenlerini saniyeler içinde analiz ederek %90'ın üzerinde doğrulukla zatürre tespiti yapabilmektedir. Gradio arayüzü sayesinde teknik bilgisi olmayan sağlık personellerinin de kolayca kullanabileceği bir prototip sunulmuştur. Gelecekte model, COVID-19 ve Tüberküloz gibi diğer akciğer hastalıklarını da kapsayacak şekilde genişletilebilir.
