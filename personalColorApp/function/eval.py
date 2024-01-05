import cv2
import glob
import numpy as np
# 준비
cascade_file = 'C:/Users/jhoon/deeplearning/cv_vs_code/cv_practice/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_file)  # 정면 얼굴 인식 모델


def automatic_brightness_adjustment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    average_brightness = cv2.mean(image, mask=threshold)[0]
    adjustment_value = 127 - average_brightness
    adjusted_image = cv2.add(image, adjustment_value)
    return adjusted_image


# 밝기 추출
def calculate_brightness(image):
    # image = detect_return_face(path)
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_brightness = np.mean(gray_image)
        return average_brightness
    else:
        return 0


# 피부색 평균 추출
def calculate_color(img):
    image = detect_return_face(img)
    if image is not None:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m = hsv_image[:, :, 0]
        avg = np.mean(m)
        return avg
    else:
        return 0


# 채도 추출
def calculate_saturation(image):
    # 이미지를 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 이미지의 채도 채널을 가져옴
    saturation_channel = hsv_image[:, :, 1]

    # 채도의 평균 계산
    average_saturation = np.mean(saturation_channel)
    return average_saturation


# 밝기 추출
def calculate_average_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_image)
    return average_brightness


# 이미지 채도 검사
def assess_colorfulness(image, saturation_threshold=30):
    if image is not None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[..., 1]
        num_high_saturation_pixels = np.sum(saturation >= saturation_threshold)
        total_pixels = np.prod(image.shape[:2])
        colorfulness_ratio = num_high_saturation_pixels / total_pixels
        return colorfulness_ratio
    else:
        return 0


# 얼굴 자르는 함수
def detect_return_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백변환
    face_list = cascade.detectMultiScale(gray, minSize=(50, 50))  # 얼굴검출

    if len(face_list) > 0:
        (x, y, w, h) = face_list[0]
        img = image[y:y + h, x:x + w]
    else:
        img = None
    return img


def detect_face_list(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백변환
    face_list = cascade.detectMultiScale(gray, minSize=(50, 50))  # 얼굴검출

    if len(face_list) > 0:
        return face_list
    else:
        return 0


# 이미지의 밝기 변화 찾아주는 함수
def func(img):
    image = detect_return_face(img)
    if image is not None:
        origin_bright = calculate_average_brightness(image)
        brightness_adjusted = automatic_brightness_adjustment(image)
        new_bright = calculate_average_brightness(brightness_adjusted)
        return origin_bright - new_bright
    else:
        print('이미지 감지실패')
        return 0


def fsm(per_list):
    # 탁기 많음 -> 회색 안어울림 가을 트루, 봄 라이트, 봄 브라이트
    if per_list[3] > 0.13:
        # 고채도 잘어울림 -> 가을트루, 봄 브라이트
        if per_list[2] > 0.85:
            if per_list[1] > 1.7:
                return "봄 브라이트"
            else:
                return "가을 트루"
        else:
            return "봄 라이트"
    # 탁기많이없음 -> 회색 어울림 가을 뮤트, 가을 딥
    else:
        if per_list[2] > 0.85:
            return "가을 딥"
        else:
            return "가을 뮤트"


def scm(per_list):
    # 밝기변화 큼 -> 여름
    if per_list[1] > 2.2:
        # 고채도 잘어울림
        if per_list[3] < 0.13:
            return "여름 뮤트"
        else:
            if per_list[2] > 0.85:
                return "여름 브라이트"
            else:
                return "여름 라이트"
    else:
        # 겨울만 남음
        if per_list[2] > 0.85:
            return "겨울 브라이트"
        else:
            return "겨울 딥"


def fsf(per_list):
    # 탁기 많음 -> 회색 안어울림 가을 트루, 봄 라이트, 봄 브라이트
    if per_list[3] > 0.13:
        # 고채도 잘어울림 -> 가을트루, 봄 브라이트
        if per_list[2] > 0.92:
            if per_list[1] > 3.4:
                return "봄 브라이트"
            else:
                return "가을 트루"
        else:
            return "봄 라이트"
    # 탁기많이없음 -> 회색 어울림 가을 뮤트, 가을 딥
    else:
        if per_list[2] > 0.92:
            return "가을 딥"
        else:
            return "가을 뮤트"


def scf(per_list):
    # 밝기변화 큼 -> 여름
    if per_list[1] > 4.8:
        # 고채도 잘어울림
        if per_list[3] < 0.13:
            return "여름 뮤트"
        else:
            if per_list[2] > 0.92:
                return "여름 브라이트"
            else:
                return "여름 라이트"
    else:
        # 겨울만 남음
        if per_list[2] > 0.92:
            return "겨울 브라이트"
        else:
            return "겨울 딥"


# 컬러, 밝기변화, 채도어울림 3개값들어있는 리스트 줌
# 일단 돌아가게는 하는 함수
def evaluate(per_list, gen):
    # 봄 겨울

    if gen == 1:
        # 원본의 얼굴밝기가 밝음
        if per_list[1] > 1.5:
            # 이와중에 보정밝기가 많이 낮음 -> 가을,봄에서 판정받기
            # 보정밝기도 낮지않은경우 -> 여기서 그대로 여름겨울로 판정
            if per_list[0] < 158:
                val = fsm(per_list)
            else:
                val = scm(per_list)
        # 원본 어두움
        else:
            if per_list[0] > 162:
                val = scm(per_list)
            else:
                val = fsm(per_list)
    else:
        if per_list[1] > 4.0:
            if per_list[0] < 162 and per_list[1] < 4.4:
                val = fsf(per_list)
            else:
                val = scf(per_list)
        else:
            if per_list[0] > 168 and per_list[0] > 3.6:
                val = scf(per_list)
            else:
                val = fsf(per_list)

    return val


# 청탁도 조사 2
def count_gray_pixels(image, x, y, w, h, lower_threshold, upper_threshold):
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_roi = gray_image[y:y + h, x:x + w]

        # 해당 영역의 픽셀 값을 가져와서 lower_threshold와 upper_threshold 사이에 있는지 확인
        mask = np.logical_and(face_roi >= lower_threshold, face_roi <= upper_threshold)
        gray_pixels = np.count_nonzero(mask)

        return gray_pixels
    else:
        return 0


# 청탁도조사 1
def detect_gray_cast(image):
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_image, minSize=(50, 50))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                gray_pixels = count_gray_pixels(image, x, y, w, h, 124, 140)
                total_pixels = w * h
                gray_ratio = gray_pixels / total_pixels
                return round(gray_ratio, 3)
        else:
            return 0.1
    else:
        return 0.1


# 얼굴안자른 청탁도 (테스트용)
def gray_cast(image):
    if image is not None:
        mask = np.logical_and(image >= 150, image <= 170)
        gray_pixels = np.count_nonzero(mask)
        height, width, _ = image.shape

        # 총 픽셀 수를 계산합니다.
        total_pixels = width * height

        return round(gray_pixels / total_pixels, 3)
    else:
        return 0.1


def non_calculate_brightness(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]  # V 채널 선택

    # 검정색에 가까운 값들을 제외한 픽셀 선택
    threshold = 30  # 조정 가능한 임계값
    selected_pixels = v_channel[v_channel > 50]

    # 선택된 픽셀의 평균 계산
    average_brightness = np.mean(selected_pixels)

    return average_brightness


# 이미지의 배경 평균밝기 조사 -> 모두 평균값으로 맞춤
# -> 얼굴만 잘라서 다시 밝기조사, 평균 알아낸후 다시 숫자정하기
def adjust_background_brightness(img, average_brightness):
    image = detect_return_face(img)
    if image is not None:
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # average_brightness = np.mean(gray_image)
        bri = non_calculate_brightness(image)
        gamma = average_brightness / bri  # 1보다 낮아지면 어두워짐
        # print("감마",gamma, '이미지 밝기', calculate_brightness(image))
        if gamma >= 1:
            gamma = (gamma - 1) / 2 + 1
        else:
            gamma = 1 - (1 - gamma) / 2
        adjusted_image = cv2.convertScaleAbs(image, alpha=gamma)
        val = non_calculate_brightness(adjusted_image)
        return val, adjusted_image
    else:
        return 0, None


def adjust_gamma(image, gamma):
    if image is not None:
        if gamma >= 1:
            gamma = (gamma - 1) / 2 + 1
        else:
            gamma = 1 - (1 - gamma) / 2
        image1 = np.uint8(255 * ((image / 255.0) ** gamma))
        return image1


# 얼굴 제외 평균밝기 계산
def calculate_background_brightness(image):
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # 얼굴 영역 제외한 영역 추출
    background = np.copy(image)
    non_face_region = 0
    for (x, y, w, h) in faces:
        background[y:y + h, x:x + w] = 128
        background[y + h:, :] = 128
        # # 선택한 영역의 평균 명암 계산
        # average_brightness1 = np.mean(non_face_region)

        # print("배경 영역의 평균 명암:", average_brightness1)
        # print(image.shape[0]*image.shape[1]/( w*h))
    # 배경 영역의 평균 밝기 계산
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = background_gray[background_gray > 40]
    average_brightness = np.mean(non_zero_pixels)
    # print('배경 평균 밝기(128)', average_brightness)
    return adjust_background_brightness(image,
                                        average_brightness)  # 높으면 밝은거, 낮으면 어두운거 -> 이미지, 밝기를 변수로 밝기 바꾸는곳에 넣음 -> 얼굴자르고 밝기 바꾼 이미지를 밝기 계산에 보냄 -> 밝기값 줌


import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모델 불러오기
loaded_model = tensorflow.keras.models.load_model("C:/Users/jhoon/deeplearning/cv_vs_code/cv_practice/gender_model1.h5")

# 데이터 전처리 및 증강 설정
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# 얼굴 자르는 함수
def detect_return_face2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백변환
    face_list = cascade.detectMultiScale(gray, minSize=(50, 50))  # 얼굴검출

    if len(face_list) > 0:
        (x, y, w, h) = face_list[0]
        img = image[y:y + h, x:x + w]
        re_img = cv2.resize(img, (100, 80))
        return re_img
    else:
        return None


def test_gender(image):
    img = detect_return_face2(image)
    if img is not None:
        image_array = np.array(img)
        # 넘파이 배열을 4차원으로 변환하여 배치 차원 추가
        image_array = np.expand_dims(image_array, axis=0)
        preprocessed_image = datagen.flow(image_array, batch_size=1, shuffle=False).next()
        prediction = loaded_model.predict(preprocessed_image)
        predicted_class = (prediction > 0.5).astype(int)  # 임계값을 기준으로 이진 분류
        return predicted_class
    return -1


# 함수들 부를 함수
def perscol(image):
    # 여기서 남녀 테스트 ㄱㄱ
    gen_val = test_gender(image)
    # ct = detect_gray_cast(image)
    val, img = calculate_background_brightness(image)
    total = [val, func(image), assess_colorfulness(img), detect_gray_cast(img)]
    if 0 in total:
        print('사진입력 에러')
        return None
    else:
        print(total)
        return evaluate(total,  gen_val)
