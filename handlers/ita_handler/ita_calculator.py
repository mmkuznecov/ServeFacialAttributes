import numpy as np
import cv2
import dlib


def extract_non_black(img):
    img_total = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    non_black = np.where(img_total > 0)
    extract_img = []
    for i in range(len(non_black[0])):
        extract_img.append(img[non_black[0][i], non_black[1][i], :])
    extract_img = np.array(extract_img).reshape(1, -1, 3)
    return extract_img


def calc_ita(masked_imgs):
    itas = []
    for masked_img in masked_imgs:
        lab_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2Lab)
        l_img, a_img, b_img = cv2.split(lab_img)
        l_img = l_img * (100 / 255)
        b_img = b_img - 128
        ITA_img = (np.arctan((l_img - 50) / b_img) * 180) / np.pi
        a_hist, a_bins = np.histogram(ITA_img, bins=1000)
        ITA_each = a_bins[a_hist.argmax()]
        itas.append(ITA_each)
    return round(sum(itas) / len(itas), 2)


class ITACalculator:

    def __init__(self, model_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = cv2.face.createFacemarkLBF()
        self.predictor.loadModel(model_path)

    def calculate_ita(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray_img)
        faces = np.empty((len(rects), 4), dtype=int)

        for i, d in enumerate(rects):
            faces[i, :] = (
                d.left(),
                d.top(),
                d.right() - d.left(),
                d.bottom() - d.top(),
            )

        ITAs = []
        for i, rect in enumerate(rects):
            img_r, img_c, img_l = self.make_masked_face_image(
                faces, i, img, img.copy(), gray_img
            )
            img_r = extract_non_black(img_r)
            img_c = extract_non_black(img_c)
            img_l = extract_non_black(img_l)
            ita = calc_ita([img_r, img_c, img_l])
            ITAs.append(ita)
        return ITAs

    def make_masked_face_image(self, faces, index, img, img_copy, gray_img):

        if not isinstance(faces, np.ndarray) or faces.dtype != int or faces.ndim != 2:
            faces = np.array(faces, dtype=int).reshape(-1, 4)

        mask_r = np.zeros_like(img)
        mask_c = np.zeros_like(img)
        mask_l = np.zeros_like(img)
        # dlib numbers
        r_cheek = [2, 3, 4, 5, 32]
        chin = [5, 6, 7, 8, 9, 10, 11, 57]
        l_cheek = [11, 12, 13, 14, 35]
        # detect landmarks
        _, landmarks = self.predictor.fit(gray_img, faces)
        landmarks = np.array(landmarks).astype(np.int32)
        for landmark in landmarks[0][0]:
            tup = (landmark[0], landmark[1])
            cv2.drawMarker(img_copy, tup, (255, 0, 0), markerSize=10, thickness=1)

        r_cheek_poly = np.empty([0, 2], int)
        l_cheek_poly = np.empty([0, 2], int)
        chin_poly = np.empty([0, 2], int)

        for mark in r_cheek:
            r_cheek_poly = np.append(
                r_cheek_poly, np.array([landmarks[index][0][mark]]), axis=0
            )
        for mark in l_cheek:
            l_cheek_poly = np.append(
                l_cheek_poly, np.array([landmarks[index][0][mark]]), axis=0
            )
        for mark in chin:
            chin_poly = np.append(
                chin_poly, np.array([landmarks[index][0][mark]]), axis=0
            )
        # create mask
        cv2.fillConvexPoly(
            mask_r, np.array(r_cheek_poly, "int32"), color=(255, 255, 255)
        )
        cv2.fillConvexPoly(
            mask_l, np.array(l_cheek_poly, "int32"), color=(255, 255, 255)
        )
        cv2.fillConvexPoly(mask_c, np.array(chin_poly, "int32"), color=(255, 255, 255))
        img_filtered = cv2.blur(img, ksize=(12, 12))

        return (
            np.where(mask_r == 255, img_filtered, mask_r),
            np.where(mask_c == 255, img_filtered, mask_c),
            np.where(mask_l == 255, img_filtered, mask_l),
        )
