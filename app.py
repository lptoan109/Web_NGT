import os
from datetime import datetime

from itsdangerous import URLSafeTimedSerializer
from flask import Flask, url_for, session, redirect, render_template, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from authlib.integrations.flask_client import OAuth
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils
import urllib.request
from werkzeug.utils import secure_filename
from ai_predictor import CoughPredictor
# --- 1. KHỞI TẠO VÀ CẤU HÌNH ---
app = Flask(__name__)

# --- THAY ĐỔI LỚN: Cấu hình từ Biến Môi Trường ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') # Render sẽ tự cung cấp biến này
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Đường dẫn lưu file sẽ trỏ đến Persistent Disk của Render
# Render sẽ gắn Disk vào đường dẫn /var/data/uploads
UPLOAD_PATH = '/var/data/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

# Cấu hình Mail từ Biến Môi Trường
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

cloudinary.config(
  cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME'), 
  api_key = os.environ.get('CLOUDINARY_API_KEY'), 
  api_secret = os.environ.get('CLOUDINARY_API_SECRET'),
  secure = True
)

# Khởi tạo các extension
db = SQLAlchemy(app)
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
bcrypt = Bcrypt(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
oauth = OAuth(app)

# --- Cấu hình Google Login ---
# !!! THAY THẾ BẰNG CLIENT ID VÀ SECRET ĐÚNG TỪ "Web client 1" !!!
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# --- TẢI MODEL AI MỘT LẦN KHI SERVER KHỞI ĐỘNG ---
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'student_model.keras')
ai_model = CoughPredictor(model_path=MODEL_FILE_PATH)

# --- 2. ĐỊNH NGHĨA MODEL ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    picture = db.Column(db.String(200), nullable=True)
    recordings = db.relationship('Recording', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Đổi filename thành audio_url
    audio_url = db.Column(db.String(300), nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- 3. ĐỊNH NGHĨA ROUTE ---

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, request.form.get('password')):
            login_user(user)
            return redirect(url_for('diagnose'))
        else:
            flash('Sai tên đăng nhập hoặc mật khẩu.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')

        existing_user_email = User.query.filter_by(email=email).first()
        if existing_user_email:
            flash('Địa chỉ email này đã được sử dụng.', 'danger')
            return redirect(url_for('register'))

        existing_user_username = User.query.filter_by(username=username).first()
        if existing_user_username:
            flash('Tên đăng nhập này đã tồn tại.', 'danger')
            return redirect(url_for('register'))

        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Tạo tài khoản thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user_info', None)
    return redirect(url_for('homepage'))

@app.route('/login/google')
def login_google():
    # Sửa http thành https ở dòng dưới
    redirect_uri = 'https://ngt.pythonanywhere.com/authorize'
    return google.authorize_redirect(redirect_uri)

@app.route('/authorize')
def authorize():
    token = google.authorize_access_token()
    user_info = google.userinfo()
    user = User.query.filter_by(email=user_info['email']).first()
    if not user:
        user = User(
            email=user_info['email'],
            username=user_info['name'],
            picture=user_info['picture']
        )
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('diagnose'))

@app.route('/diagnose')
def diagnose():
    # Route này không cần login, ai cũng có thể truy cập trang
    return render_template('diagnose.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            sender_email_from_form = request.form.get('email')
            subject = request.form.get('subject')
            message_body = request.form.get('message')

            msg = Message(
                subject=f"Tin nhắn từ Web NGT Cough: {subject}",
                sender=("Website NGT Cough", app.config['MAIL_USERNAME']),
                recipients=[app.config['MAIL_USERNAME']]
            )
            msg.body = f"Message from: {name} <{sender_email_from_form}>\n\n{message_body}"
            mail.send(msg)
            flash('Cảm ơn bạn đã gửi tin nhắn! Chúng tôi sẽ phản hồi sớm.', 'success')
        except Exception as e:
            flash('Đã có lỗi xảy ra khi gửi tin nhắn. Vui lòng thử lại.', 'danger')
            print(e)
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    pagination = Recording.query.filter_by(user_id=current_user.id)\
                                .order_by(Recording.timestamp.desc())\
                                .paginate(page=page, per_page=10, error_out=False)
    recordings = pagination.items
    return render_template('history.html', recordings=recordings, pagination=pagination)

@app.route('/delete_recording/<int:recording_id>', methods=['POST'])
@login_required
def delete_recording(recording_id):
    recording = Recording.query.get_or_404(recording_id)
    if recording.user_id != current_user.id:
        return {"error": "Unauthorized"}, 403
    try:
        filepath = os.path.join(app.root_path, 'uploads', recording.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        db.session.delete(recording)
        db.session.commit()
        flash('Đã xóa bản ghi thành công.', 'success')
    except Exception as e:
        flash('Đã có lỗi xảy ra khi xóa file.', 'danger')
        print(f"Error deleting file: {e}")
    return redirect(url_for('history'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        new_username = request.form.get('username')
        current_user.username = new_username
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
                os.makedirs(upload_path, exist_ok=True)
                file.save(os.path.join(upload_path, filename))
                current_user.picture = f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        db.session.commit()
        flash('Cập nhật thông tin thành công!', 'success')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files.get('audio_data')
    if not audio_file:
        return jsonify({"error": "Không có file âm thanh"}), 400

    try:
        # 1. Tải file gốc lên Cloudinary
        upload_result = cloudinary.uploader.upload(
            audio_file,
            resource_type="video",
            folder="cough_audio"
        )
        
        # --- THAY ĐỔI LỚN: Dùng hàm chính thức để tạo URL ---
        public_id = upload_result['public_id']

        # Yêu cầu thư viện Cloudinary tạo URL chuyển đổi giúp chúng ta
        # Điều này đảm bảo URL luôn đúng cú pháp.
        transformed_url = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type="video",
            transformation=[
                {'audio_codec': 'none'}, # Cần thiết cho việc đổi sample rate
                {'audio_frequency': 16000},
                {'audio_channels': 'mono'}
            ]
        )[0] # Lấy URL từ kết quả trả về

        # Đảm bảo URL kết thúc bằng .wav
        # Hàm trên có thể trả về url không có đuôi, ta thêm vào cho chắc
        if not transformed_url.endswith('.wav'):
             transformed_url = transformed_url.rsplit('.', 1)[0] + ".wav"
        # -----------------------------------------------------------

        # 2. Tải file đã được chuyển đổi hoàn toàn về server
        temp_filename = "temp_audio_for_ai.wav"
        urllib.request.urlretrieve(transformed_url, temp_filename)

        # 3. Gọi model AI
        ai_result = ai_model.predict(temp_filename)

        # 4. Xóa file tạm
        os.remove(temp_filename)

        # 5. Lưu vào database
        if current_user.is_authenticated and 'error' not in ai_result:
            new_prediction = Prediction(
                user_id=current_user.id,
                audio_url=upload_result['secure_url'],
                result=ai_result.get('predicted_class', 'N/A'),
                confidence=ai_result.get('confidence', '0%')
            )
            db.session.add(new_prediction)
            db.session.commit()
        
        # 6. Trả kết quả về
        return jsonify({"success": True, "diagnosis_result": ai_result})

    except Exception as e:
        print(f"Lỗi khi tải lên hoặc dự đoán: {e}")
        return jsonify({"error": "Lỗi máy chủ trong quá trình xử lý"}), 500

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

# app.py

# --- HÀM TẠO TOKEN BẢO MẬT ---
def generate_reset_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt='password-reset-salt')

# --- HÀM XÁC THỰC TOKEN ---
def confirm_reset_token(token, expiration=3600): # Hết hạn sau 1 giờ
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
        return email
    except Exception:
        return False

# --- ROUTE XỬ LÝ QUÊN MẬT KHẨU ---
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            token = generate_reset_token(user.email)
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Soạn và gửi email
            msg = Message('Yêu cầu đặt lại mật khẩu - NGT Cough',
                          sender=("Website NGT Cough", app.config['MAIL_USERNAME']),
                          recipients=[user.email])
            msg.body = f'''Chào {user.username},

Để đặt lại mật khẩu của bạn, vui lòng nhấn vào đường link sau:
{reset_url}

Nếu bạn không phải là người yêu cầu, vui lòng bỏ qua email này. Link sẽ hết hạn sau 1 giờ.

Trân trọng,
Đội ngũ NGT Cough'''
            mail.send(msg)
            
        # Dù email có tồn tại hay không, vẫn hiện thông báo này để bảo mật
        flash('Nếu email của bạn tồn tại trong hệ thống, một hướng dẫn đặt lại mật khẩu đã được gửi đến.', 'success')
        return redirect(url_for('login'))
        
    return render_template('forgot_password.html')


# --- ROUTE XỬ LÝ ĐẶT LẠI MẬT KHẨU ---
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = confirm_reset_token(token)
    if not email:
        flash('Đường link đặt lại mật khẩu không hợp lệ hoặc đã hết hạn.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
        
        user = User.query.filter_by(email=email).first()
        user.password_hash = hashed_password
        db.session.commit()
        
        flash('Mật khẩu của bạn đã được cập nhật thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)