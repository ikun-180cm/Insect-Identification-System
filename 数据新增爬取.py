import os
import time
import requests
import urllib3
from tqdm import tqdm
import random
from requests.exceptions import RequestException, JSONDecodeError
from PIL import Image
import io
import cv2
import numpy as np

# 禁用SSL警告
urllib3.disable_warnings()

# ===================== 固定配置 =====================
BASE_DIR = "昆虫_split"
CLASS_LIST = [
    "Bees", "Beetles", "Butterfly", "Cicada", "Dragonfly",
    "Grasshopper", "Moth", "Scorpion", "Snail", "Spider"
]

SEARCH_WORDS = {
    "Bees": "蜜蜂", "Beetles": "甲虫", "Butterfly": "蝴蝶", "Cicada": "蝉",
    "Dragonfly": "蜻蜓", "Grasshopper": "蚱蜢", "Moth": "飞蛾",
    "Scorpion": "蝎子", "Snail": "蜗牛", "Spider": "蜘蛛"
}

# 昆虫特征颜色范围（用于检测图片中是否真的存在昆虫）
INSECT_COLOR_RANGES = {
    "Bees": [(0, 100, 100), (30, 255, 255)],  # 黄/黑
    "Beetles": [(0, 0, 0), (180, 255, 50)],   # 黑壳
    "Butterfly": [(80, 100, 100), (130, 255, 255)],  # 彩色
    "Cicada": [(20, 100, 100), (40, 255, 200)],
    "Dragonfly": [(90, 100, 100), (130, 255, 255)],
    "Grasshopper": [(30, 80, 80), (80, 255, 200)],
    "Moth": [(0, 0, 50), (180, 50, 200)],
    "Scorpion": [(0, 0, 0), (180, 255, 80)],
    "Snail": [(10, 80, 100), (30, 255, 200)],
    "Spider": [(0, 0, 0), (180, 255, 100)]
}

# 新图片固定从 1000 开始命名
START_INDEX = 1000

# 你设定的最终数量
TRAIN_TOTAL = 1100    # 训练集每类总目标
TEST_TOTAL = 800     # 测试集每类总目标

# 广告/垃圾图过滤关键词
AD_FILTER = [
    "搞","优惠券","秒杀","活动",
    "专属","爆款","包邮","直播",
    "窝","女","男"
]

CONFIG = {
    "request_timeout": 10,
    "retry_times": 3,
    "sleep_range": (0.3, 0.7),
    "min_size": 200,  # 图片最小像素
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36",
    ]
}

# 老师原版 Cookies
cookies = {
    'BDqhfp': '%E7%8B%97%E7%8B%97%26%26NaN-undefined%26%2618880%26%2621',
    'BIDUPSID': '06338E0BE23C6ADB52165ACEB972355B',
    'PSTM': '1646905430',
    'BAIDUID': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
    'H_PS_PSSID': '35836_35105_31254_36024_36005_34584_36142_36120_36032_35993_35984_35319_26350_35723_22160_36061',
    'BDSFRCVID': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
    'H_BDCLCKID_SF': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRDInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'BDSFRCVID_BFESS': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
    'H_BDCLCKID_SF_BFESS': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRDInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
    'indexPageSugList': '%5B%22%E7%8B%97%E7%8B%97%22%5D',
    'cleanHistoryStatus': '0',
    'BAIDUID_BFESS': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
    'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
    'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
    'ab_sr': '1.0.1_Y2YxZDkwMWZkMmY2MzA4MGU0OTNhMzVlNTcwMmM2MWE4YWU4OTc1ZjZmZDM2N2RjYmVkMzFiY2NjNWM4Nzk4NzBlZTliYWU0ZTAyODkzNDA3YzNiMTVjMTllMzQ0MGJlZjAwYzk5MDdjNWM0MzJmMDdhOWNhYTZhMjIwODc5MDMxN2QyMmE1YTFmN2QyY2M1M2VmZDkzMjMyOThiYmNhZA==',
    'delPer': '0',
    'PSINO': '2',
    'BA_HECTOR': '824a024042g05alup1h3g0aq0q',
}

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_random_headers():
    headers = {
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
        'Accept': 'text/plain, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': random.choice(CONFIG["user_agents"]),
        'sec-ch-ua-platform': '"Windows"',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://image.baidu.com/',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    return headers

# ===================== 检测图片中是否有人脸 =====================
def has_human(img_data):
    try:
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        return len(faces) > 0
    except:
        return True

# ===================== 检测图片是否真的包含目标昆虫 =====================
def has_target_insect(img_data, insect_class):
    try:
        img = Image.open(io.BytesIO(img_data))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        lower, upper = INSECT_COLOR_RANGES[insect_class]
        mask = cv2.inRange(img_cv, np.array(lower), np.array(upper))
        ratio = cv2.countNonZero(mask) / (img_cv.size / 3)
        return ratio > 0.02
    except:
        return False

# ===================== 过滤广告/无效图/无昆虫图 =====================
def is_valid_image(img_url, img_data, insect_class):
    try:
        # 1. 过滤广告关键词
        for kw in AD_FILTER:
            if kw in img_url:
                return False

        # 2. 过滤尺寸过小
        img = Image.open(io.BytesIO(img_data))
        w, h = img.size
        if w < CONFIG["min_size"] or h < CONFIG["min_size"]:
            return False

        # 3. 过滤有人的图片
        if has_human(img_data):
            return False

        # 4. 过滤没有目标昆虫的图片
        if not has_target_insect(img_data, insect_class):
            return False

        return True
    except:
        return False

def download_image(img_url, save_path, insect_class, retry=0):
    if retry >= CONFIG["retry_times"]:
        return False
    try:
        time.sleep(random.uniform(*CONFIG["sleep_range"]))
        resp = requests.get(
            img_url, verify=False, timeout=CONFIG["request_timeout"],
            headers={"User-Agent": random.choice(CONFIG["user_agents"])}
        )
        resp.raise_for_status()

        if not is_valid_image(img_url, resp.content, insect_class):
            return False

        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True
    except RequestException:
        return download_image(img_url, save_path, insect_class, retry + 1)

# ===================== 统计文件夹图片数量（已修复） =====================
def count_images(folder):
    if not os.path.exists(folder):
        return 0
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    cnt = 0
    for f in os.listdir(folder):
        if f.endswith(exts):
            cnt += 1
    return cnt

# ===================== 核心爬取函数（带检测跳过） =====================
def craw_class(keyword_en, target_total, is_train=True):
    sub_dir = "train" if is_train else "test"
    save_dir = os.path.join(BASE_DIR, sub_dir, keyword_en)
    os.makedirs(save_dir, exist_ok=True)

    current_count = count_images(save_dir)
    print(f"\n[{keyword_en}] {sub_dir} 已有: {current_count}/{target_total}")

    if current_count >= target_total:
        print(f"✅ 数量已满，自动跳过！")
        return

    need_download = target_total - current_count
    print(f"➡ 需要下载: {need_download} 张")

    current_idx = START_INDEX
    downloaded = 0

    pbar = tqdm(total=need_download, desc=f"爬取 {keyword_en}")
    page = 1
    search_word = SEARCH_WORDS[keyword_en]

    while downloaded < need_download:
        pn = 30 * page
        params = (
            ('tn', 'resultjson_com'), ('ipn', 'rj'), ('ct', '201326592'),
            ('word', search_word), ('queryWord', search_word),
            ('cl', '2'), ('lm', '-1'), ('ie', 'utf-8'), ('oe', 'utf-8'),
            ('pn', str(pn)), ('rn', '30'), ('nc', '1'),
        )

        try:
            res = requests.get(
                'https://image.baidu.com/search/acjson',
                headers=get_random_headers(), params=params, cookies=cookies,
                verify=False, timeout=CONFIG["request_timeout"]
            )
            items = res.json().get("data", [])

            for item in items:
                if downloaded >= need_download:
                    break

                img_url = item.get("thumbURL")
                if not img_url:
                    continue

                filename = f"{current_idx:08d}.jpg"
                save_path = os.path.join(save_dir, filename)

                if download_image(img_url, save_path, keyword_en):
                    downloaded += 1
                    current_idx += 1
                    pbar.update(1)

            page += 1
        except:
            page += 1

    pbar.close()
    print(f"✅ {keyword_en} 完成！\n")

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("    昆虫数据集爬虫（自动检测数量 + 过滤广告 + 昆虫内容验证）")
    print("=" * 60)

    # 训练集
    print("\n📌 开始爬取：训练集 train（每类目标 3500 张）")
    for cls in CLASS_LIST:
        craw_class(cls, TRAIN_TOTAL, is_train=True)
        time.sleep(random.uniform(0.5, 0.9))

    # 测试集
    print("\n📌 开始爬取：测试集 test（每类目标 1500 张）")
    for cls in CLASS_LIST:
        craw_class(cls, TEST_TOTAL, is_train=False)
        time.sleep(random.uniform(0.5, 0.9))

    print("\n🎉 全部爬取完成！")

