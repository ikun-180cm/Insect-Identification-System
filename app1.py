import os.path
from fileinput import filename
import cv2
import json
from flask import Flask, render_template, request, redirect, session
import sqlite3
from datetime import datetime
# 新增模型相关依赖
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = "123456"

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 上传文件保存目录
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 最大允许 100MB 文件
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', }#'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm'

# ====================== 模型配置 ======================
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import os
import timm

# 昆虫中文名称映射
CLASS_NAMES = ['Bees', 'Beetles', 'Butterfly', 'Cicada', 'Dragonfly',
               'Grasshopper', 'Moth', 'Scorpion', 'Snail', 'Spider']

CLASS_CHINESE = [
    "蜜蜂",
    "甲虫",
    "蝴蝶",
    "蝉",
    "蜻蜓",
    "蚱蜢",
    "飞蛾",
    "蝎子",
    "蜗牛",
    "蜘蛛"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=10)
    model.load_state_dict(torch.load('swin_best_final.pth', map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ======================优化==================
def predict_image(image_path):
    try:
        # 1. 打开原图
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # 2. 模型推理
        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = MODEL(x)
            pred_idx = output.argmax(1).item()

        # 3. 英文 → 中文
        class_en = CLASS_NAMES[pred_idx]
        class_cn = CLASS_CHINESE[pred_idx]

        # 4. 只保留方框圈选（恢复原始功能）
        draw = ImageDraw.Draw(image)
        box_margin = 0.1
        x1 = int(w * box_margin)
        y1 = int(h * box_margin)
        x2 = int(w * (1 - box_margin))
        y2 = int(h * (1 - box_margin))
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)

        # 5. 结果图保存
        result_filename = os.path.basename(image_path)
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        image.save(result_path)

        return f"{class_cn} ({class_en})", result_filename

    except Exception as e:
        print("==================== 检测失败真实原因 ====================")
        print(f"图片路径: {image_path}")
        print(f"错误信息: {str(e)}")
        print("==========================================================")
        return "识别失败", None


# ======================数据库======================
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def get_db():
    return sqlite3.connect("昆虫检测数据库.db")


def init_db():  # 数据库初始化
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    """)
    # 改造record表：新增detect_result字段存储检测结果
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS record (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        filename TEXT,
        upload_time TEXT,
        detect_result TEXT,
        result_filename TEXT  -- 新增：结果图文件名
    )
    """)
    conn.commit()
    conn.close()


#==================接口==================
@app.route('/')
def index():
    if 'username' not in session:
        return redirect('/login')
    return render_template("index.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user WHERE username=? AND password=?",
            (username, password)
        )
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect('/')
        else:
            return render_template("error.html",
                                   message="用户名或密码错误，请重试",
                                   btn_text="返回登录处",
                                   back_url="/login")
    return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM user WHERE username=?",
            (username,)
        )
        if cursor.fetchone():
            conn.close()
            return render_template("error.html",
                                   message="用户名已存在，请重新注册",
                                   btn_text="返回注册处",
                                   back_url="/register")
        cursor.execute(
            "INSERT INTO user (username,password) VALUES (?,?)",
            (username, password)
        )
        conn.commit()
        conn.close()

        return redirect('/login')
    return render_template("register.html")


@app.route('/error')
def error():
    message = request.args.get('msg', '用户名或密码错误，请重试')
    return render_template('error.html', message=message)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 文件上传
@app.route('/upload', methods=['POST'])
def upload():
    if 'username' not in session:
        return redirect('/login')

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template("error.html", message="未选择文件", btn_text="返回首页", back_url="/")

    if not allowed_file(file.filename):
        return render_template("error.html", message="不支持的文件格式", btn_text="返回首页", back_url="/")

    # 防重名覆盖
    filename = file.filename
    base, ext = os.path.splitext(filename)
    new_filename = filename
    counter = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    try:
        # 保存原图
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(file_path)

        #判断格式
        ext_lower = ext.lower().lstrip('.')
        if ext_lower in ['jpg', 'jpeg', 'png', 'bmp']:
            detect_result, result_filename = predict_image(file_path)
        else:
            detect_result = "不支持该格式检测"
            result_filename = None

        #存入数据库
        conn = get_db()
        cursor = conn.cursor()
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO record 
            (username, filename, upload_time, detect_result, result_filename) 
            VALUES (?,?,?,?,?)
            """,
            (session['username'], new_filename, upload_time, detect_result, result_filename)
        )
        conn.commit()
        print(f"【DEBUG】检测结果: {detect_result}, 结果图文件名: {result_filename}")
    except Exception as e:
        return render_template("error.html", message=f"上传失败：{str(e)}", btn_text="返回首页", back_url="/")
    finally:
        if 'conn' in locals():
            conn.close()

    return redirect(f'/result?filename={new_filename}&detect_result={detect_result}')

# 结果页
@app.route('/result')
def result():
    if 'username' not in session:
        return redirect('/login')
    filename = request.args.get('filename')
    detect_result = request.args.get('detect_result', '未知')
    return render_template('result.html', filename=filename, detect_result=detect_result)


# 历史记录
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect('/login')

    # 获取当前页码，默认第1页
    page = request.args.get('page', 1, type=int)
    per_page = 5  # 每页显示5条
    offset = (page - 1) * per_page

    try:
        conn = get_db()
        cursor = conn.cursor()

        # 查询总记录数
        cursor.execute("""
            SELECT COUNT(*) FROM record
            WHERE username = ?
        """, (session['username'],))
        total = cursor.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page

        #查询所有字段，拿到 result_filename
        cursor.execute("""
            SELECT id, username, filename, upload_time, detect_result, result_filename
            FROM record
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (session['username'], per_page, offset))

        records = cursor.fetchall()

        #不依赖数据库存储的 result_filename
        final_records = []
        for record in records:
            # record = (id, username, filename, upload_time, detect_result, result_filename)
            filename = record[2]
            # 用原图文件名作为结果图文件名
            result_filename = filename
            final_records.append( (record[0], record[1], record[2], record[3], record[4], result_filename) )

    except sqlite3.Error as e:
        return render_template("error.html",
                               message=f"获取历史记录失败: {e}",
                               btn_text="返回首页",
                               back_url="/")
    finally:
        conn.close()

    return render_template(
        "history.html",
        records=final_records,
        page=page,
        total_pages=total_pages
    )


if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')



@app.route('/delete', methods=['POST'])
def delete():
    if 'username' not in session:
        return redirect('/login')
    record_id = request.form.get('id')
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename FROM record WHERE id=? AND username=?",
        (record_id, session['username'])
    )
    result = cursor.fetchone()
    if result:
        filename = result[0]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        cursor.execute(
            "DELETE FROM record WHERE id=? AND username=?",
            (record_id, session['username'])
        )
        conn.commit()
    conn.close()
    return redirect('/history')


if __name__ == "__main__":
    init_db()
    app.run(debug=True)