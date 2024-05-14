import cv2
import numpy as np
import sys
import os

def circle_detect(img_path):
    '''
    Bu kod, terminal üzerinden circle detection dosyasını çalıştırmak için yazılmıştır.
    
    dp=1, #dp: çözünürlük oranı.
    minDist=200, #minDist: algılanabilecek daireler arasındaki minimum mesafe.
    param1=255, # param1: Canny Edge Detector için üst eşik değeri.
    param2=20, # param2: dairesel merkezlerin tespiti için eşik değeri.
    minRadius=50, # minRadius: algılanacak dairelerin minimum yarıçapı.
    maxRadius=5000 # maxRadius: algılanacak dairelerin maksimum yarıçapı.
    '''

    image_path = img_path # Görselin Path'ini alıyoruz.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) # Path üzerinden görseli okuyoruz.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Görseli Grayscale olarak convert ediyoruz.
    edges = cv2.Canny(gray, threshold1=254, threshold2=255) # Edge Detection için Canny yöntemini kullanıyoruz. threshold1: kenar olarak kabul edilen zayıf piksel değerlerinin alt sınırı, threshold2: kenar olarak kabul edilen güçlü piksel değerlerinin üst sınırı.

    # Morfolojik işlemler uygulayarak bir maskeye dönüştürüyoruz.
    # kernel: morfolojik işlemler için kullanılan kernel, burada 5x5 boyutunda bir matris. Çünkü yüksek çözünürlüklü görseller için 5x5 matris kullanabiliriz. Yoksa 3x3 kullanmalıydık.
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4) # cv2.MORPH_CLOSE, morfolojik kapanma işlemi olarak bilinir. Kenarların içerisindeki küçük boşlukları doldurur. iterations: işlemi kaç kez tekrarlayacağını belirler, burada 4 kez tekrarlıyoruz.

    circles = cv2.HoughCircles( #Hough Dönüşümü ile daireleri algılıyoruz. 
        closed_mask, 
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200, 
        param1=255,
        param2=20,
        minRadius=50,
        maxRadius=5000
    )

    if circles is not None: # Eğer daireler algılandıysa, her bir daireyi çiziyoruz ve yarıçap bilgisini yazdırıyoruz.
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            radius_info = f"Radius: {circle[2]}"
            text_size = cv2.getTextSize(radius_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = circle[0] - text_size[0] // 2
            text_y = circle[1] + circle[2] + text_size[1] + 5
            cv2.putText(image, radius_info, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Sonunda, detect edilen daireler ile görüntüyü kaydediyoruz ve bununla ilgili kullanıcıyı bilgilendiriyoruz.
    output_path = os.path.splitext(image_path)[0] + '_output.jpg'
    cv2.imwrite(output_path, image)
    cv2.imshow("Result", image)
    print(f"Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Result image saved at: {output_path}")


def main():
    if (len(sys.argv) < 2):
        print("Usage of this code: python3 circle_detection.py <image_path>")
        sys.exit(1)
    else:
        if len(sys.argv[1]) <= 2:
            print("Usage of this code: python3 circle_detection.py <image_path>")
            sys.exit(1)
        else:
            image = sys.argv[1]
            circle_detect(image)


    if len(sys.argv) < 2:
        print("Kullanım: python3 circle_detection.py <label_folder>")
        sys.exit(1)

    

if __name__ == "__main__":
    main()