# Submission 1: NPL-Disaster-Tweets
Nama: Imam Agus Faisal

Username dicoding: imamaf

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [NLP with Disaster Tweets - cleaning data](https://www.kaggle.com/datasets/vbmokin/nlp-with-disaster-tweets-cleaning-data) |
| Masalah | Bencana alam seperti gempa bumi, banjir, kebakaran hutan, atau badai tropis sering kali memicu respons cepat dari masyarakat dan pihak berwenang. Media sosial, terutama Twitter, telah menjadi sumber utama informasi dan komunikasi selama bencana alam. Namun, masalah muncul ketika informasi yang beredar di Twitter tidak dapat dipercaya dan dapat berpotensi memicu kepanikan atau tindakan yang tidak sesuai. Hal ini sering kali disebabkan oleh tersebarnya hoax atau informasi palsu yang disamarkan sebagai berita terkini tentang bencana alam.|
| Solusi machine learning | Masalah ini mengharuskan pengembangan algoritma NLP (*Natural Language Processing*) dengan menggunakan *machine learning* untuk mengidentifikasi dan mengatasi *tweets* hoax yang berkaitan dengan bencana alam. |
| Metode pengolahan | Metode pengolahan data yang digunakan pada proyek ini berupa mengubah text menjadi susunan angka dengan tokenisasi input yang merepresentasikan text tersebut agar mudah dimengerti oleh model. |
| Arsitektur model | Model dibangun dengan menggunakan arsitektur layer TextVectorization sebagai layer yang memproses input teks atau string kedalam bentuk susunan angka (representasi numerik). Lalu data tersebut akan diproses pada layer Embedding untuk mempelajari kemiripan atau kedekatan dari sebuah kata agar dapat diketahui bahwa kata tersebut merupakan kata positif atau negatif (hoax). Kemudian, terdapat juga 2 *hidden layer* dan 1 *output layer*. |
| Metrik evaluasi | Menggunakan metrik Binary Accuracy, False Negatives, False Positive, True Negatives, dan True Positive. |
| Performa model | Performa dari model yang dibuat sudah menunjukkan hasil yang baik. Model dapat mengklasifikasikan Tweets bencana alam dengan tepat, yaitu Tweet tersebut termasuk kedalam Tweet bencana alam asli atau hoax |