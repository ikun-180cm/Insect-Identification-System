import os
import requests
import time
import random
from PIL import Image
from tqdm import tqdm

# ====================== 配置 ======================
BASE_DIR = "昆虫_split"
CLASSES = ["Bees", "Beetles", "Butterfly", "Cicada", "Dragonfly", "Grasshopper", "Moth", "Scorpion", "Snail", "Spider"]

TARGETS = {"train": 1100, "test": 800}

# 改进后的查询配置（强烈推荐使用 taxon_id）
QUERY_MAP = {
    "Bees": "bees",                    # 保留备用
    "Beetles": "beetles",
    "Butterfly": "butterfly",
    "Cicada": "cicada",
    "Dragonfly": "dragonfly",
    "Grasshopper": "grasshopper OR orthoptera",
    "Moth": "moth",
    "Scorpion": "scorpion",
    "Snail": "snail",
    "Spider": "spider",
}

TAXON_ID_MAP = {
    "Bees": 47221,      # Apidae（蜜蜂科，包括 honeybee、bumblebee 等）——这是最合适的宽泛蜜蜂类
    # "Bees": 630955,   # 如果你只想要 Bombus（熊蜂属），可以换成这个
    "Beetles": 47208,    # Coleoptera（甲虫目）
    "Butterfly": 47157, # Lepidoptera 中的 butterflies（蝴蝶）
    "Cicada": 47744,    # Cicadidae 或更广的 Cicadoidea
    "Dragonfly": 47744, # Odonata（蜻蜓目）
    "Grasshopper": 47158, # Orthoptera（直翅目）
    "Moth": 47157,      # Lepidoptera（可再细分）
    "Scorpion": 47215,  # Scorpiones（蝎目）
    "Snail": 47119,     # Gastropoda（腹足纲，包含 snail）
    "Spider": 47118,    # Araneae（蜘蛛目）
}
# ====================== 辅助函数 ======================
def get_image_count(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def get_next_image_name(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    nums = []
    for f in files:
        try:
            num = int(os.path.splitext(f)[0])
            nums.append(num)
        except:
            continue
    max_num = max(nums) if nums else 0
    next_num = max(max_num + 1, 10000)
    return f"{next_num:08d}.jpg"


def fetch_inat_observations(cls, num_needed, page=1, per_page=200):
    url = "https://api.inaturalist.org/v1/observations"

    params = {
        "photos": "true",
        "order_by": "observed_on",  # 按观察日期，能拿到更多历史数据
        "order": "desc",
        "per_page": per_page,
        "page": page,
        "quality_grade": "research,needs_id",  # 优先高质量
        "locale": "zh-CN"
    }

    # ============ 加上地理限制 ============
    # 方式一：使用 place_id（推荐，精确且高效）
    params["place_id"] = "6903"  # 中国大陆
    # params["place_id"] = "8525"        # 香港（如果你只想香港数据）
    # params["place_id"] = "6903,8525"   # 中国+香港

    # 方式二：如果想用经纬度范围（bounding box），用下面这行替换上面 place_id
    # params["nelat"] = 53.6      # 东北角纬度（中国大致范围）
    # params["nelng"] = 134.0     # 东北角经度
    # params["swlat"] = 18.0      # 西南角纬度
    # params["swlng"] = 73.5      # 西南角经度

    # ============ taxon 处理 ============
    taxon_id = TAXON_ID_MAP.get(cls)
    if taxon_id:
        params["taxon_id"] = taxon_id
        print(f"  使用 taxon_id={taxon_id} + 中国地区 搜索 {cls}")
    else:
        taxon_name = QUERY_MAP.get(cls, cls)
        params["taxon_name"] = taxon_name
        print(f"  使用 taxon_name='{taxon_name}' + 中国地区 搜索 {cls}")

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 429:
            print("⚠️  触发速率限制！请等待几分钟后再试...")
            time.sleep(60)
            return [], 0
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        total = data.get("total_results", 0)
        print(f"  第 {page} 页 获取到 {len(results)} 条记录（总计约 {total} 条）")
        return results, total
    except Exception as e:
        print(f"API 请求失败: {e}")
        return [], 0


def download_photo(photo_url, save_path, min_size=300):
    """下载并验证照片"""
    try:
        resp = requests.get(photo_url, timeout=20, stream=True)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        with Image.open(save_path) as img:
            img.verify()
            w, h = img.size
            if w < min_size or h < min_size:
                os.remove(save_path)
                return False
        return True
    except:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


# ====================== 主程序 ======================
if __name__ == "__main__":
    for cls in CLASSES:
        train_folder = os.path.join(BASE_DIR, "train", cls)
        test_folder = os.path.join(BASE_DIR, "test", cls)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        curr_train = get_image_count(train_folder)
        curr_test = get_image_count(test_folder)

        need_train = max(0, TARGETS["train"] - curr_train)
        need_test = max(0, TARGETS["test"] - curr_test)

        if need_train == 0 and need_test == 0:
            print(f"✅ {cls} 已达到目标数量，跳过。")
            continue

        print(f"\n🚀 开始处理 {cls} | 训练集还需 {need_train} 张 | 测试集还需 {need_test} 张")

        taxon_name = QUERY_MAP.get(cls, cls)
        page = 1
        added_train = added_test = 0
        total_needed = need_train + need_test

        while (added_train + added_test) < total_needed * 1.2:   # 多抓一点备用
            observations, total = fetch_inat_observations(taxon_name, total_needed, page=page)
            if not observations:
                print("  没有更多观察记录了")
                break

            print(f"  第 {page} 页 获取到 {len(observations)} 条观察记录")

            for obs in observations:
                if not obs.get("photos"):
                    continue
                # 优先取最大尺寸照片（original > large > medium）
                photo = obs["photos"][0]
                photo_url = photo.get("url")
                if not photo_url:
                    continue
                # 替换尺寸为 original（最高清）
                photo_url = photo_url.replace("square", "original").replace("medium", "original").replace("large", "original")

                if added_train < need_train:
                    filename = get_next_image_name(train_folder)
                    save_path = os.path.join(train_folder, filename)
                    if download_photo(photo_url, save_path):
                        added_train += 1
                        print(f"   训练集 ✓ {filename}")
                elif added_test < need_test:
                    filename = get_next_image_name(test_folder)
                    save_path = os.path.join(test_folder, filename)
                    if download_photo(photo_url, save_path):
                        added_test += 1
                        print(f"   测试集 ✓ {filename}")

                if added_train >= need_train and added_test >= need_test:
                    break

                time.sleep(random.uniform(0.8, 1.5))   # 重要：礼貌延时

            page += 1
            time.sleep(random.uniform(1.5, 3.0))

        print(f"🎉 {cls} 处理完成！训练集新增 {added_train} 张，测试集新增 {added_test} 张\n")

    print("✅ 全部类别处理完毕！建议分几天运行，避免被限流。")