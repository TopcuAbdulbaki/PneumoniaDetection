X-Ray Görüntülerinden Zatürre Tespiti
Bu projede, akciğer röntgen (X-Ray) görüntülerini analiz ederek hastanın Zatürre (Pneumonia) olup olmadığını tespit eden bir derin öğrenme modeli geliştirilmiştir.

Model, Transfer Learning yöntemi kullanılarak VGG16 mimarisi üzerinde eğitilmiştir. Kaggle'daki "Chest X-Ray Images" veri seti kullanılarak eğitilen model, test verileri üzerinde %86 üzeri doğruluk oranına ulaşmıştır. Ayrıca modelin herkes tarafından kolayca kullanılabilmesi için Gradio tabanlı sürükle-bırak destekli bir web arayüzü tasarlanmıştır.

Proje kapsamında kullanıldı:
Python (3.10+)

TensorFlow / Keras (Derin Öğrenme Modeli)

OpenCV (Görüntü İşleme)

NumPy (Veri Analizi)

Gradio (Kullanıcı Arayüzü)

🚀 Kurulum ve Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1. Gerekli Kütüphaneleri Yükleyin
Proje dizininde bir terminal açın ve bağımlılıkları yükleyin:

Bash

pip install -r requirements.txt
2. Uygulamayı Başlatın
Gradio arayüzünü başlatmak için aşağıdaki komutu çalıştırın:

Bash

python app.py
Komutu çalıştırdıktan sonra terminalde verilen linke (örneğin: http://127.0.0.1:7860) tıklayarak uygulamayı tarayıcınızda kullanabilirsiniz.


Denemek için: https://huggingface.co/spaces/Adam3438/PneumoniaDetection

