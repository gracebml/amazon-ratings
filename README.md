## 1. Pure NumPy Recommender – Project Overview

Feature-enriched collaborative filtering pipeline for the Amazon Beauty ratings dataset, implemented end-to-end with NumPy (no pandas/scikit-learn during modeling).

---

## 2. Mục lục

- [1. Pure NumPy Recommender – Project Overview](#1-pure-numpy-recommender--project-overview)
- [2. Mục lục](#2-mục-lục)
- [3. Giới thiệu](#3-giới-thiệu)
- [4. Dataset](#4-dataset)
- [5. Method](#5-method)
  - [5.1 Quy trình xử lý dữ liệu](#51-quy-trình-xử-lý-dữ-liệu)
  - [5.2 Phương pháp: Collaborative Filtering Matrix Factorization](#52-phương-pháp-Collaborative-Filtering-Matrix-Factorization)
  - [5.3 Triển khai thuần NumPy](#53-triển-khai-thuần-numpy)
- [6. Installation \& Setup](#6-installation--setup)
- [7. Usage](#7-usage)
- [8. Results](#8-results)
- [9. Project Structure](#9-project-structure)
- [10. Challenges \& Solutions](#10-challenges--solutions)
- [11. Future Improvements](#11-future-improvements)
- [12. Contributors](#12-contributors)
- [13. License](#13-license)

---

## 3. Giới thiệu

- **Bài toán**: Xây dựng hệ thống gợi ý sản phẩm làm đẹp (Beauty) dựa trên rating 1–5 sao của người dùng Amazon.
- **Động lực & ứng dụng**:
  - Giúp người dùng tìm sản phẩm phù hợp nhanh hơn.
  - Giảm churn cho nền tảng thương mại điện tử nhờ đề xuất cá nhân hóa.
  - Cung cấp baseline NumPy-only cho học phần *Numpy for Data Science*.
- **Mục tiêu cụ thể**:
  1. Làm sạch & xác thực dữ liệu thô 2M dòng.
  2. Kỹ thuật đặc trưng (feature engineering) cho user & item để cải thiện cold-start.
  3. Triển khai Matrix Factorization thuần NumPy với SGD, gradient clipping và regularization.
  4. Đánh giá bằng RMSE/MAE, phân tích lỗi & insight.

---

## 4. Dataset

- **Nguồn**: Amazon Product Data – Beauty (Julian McAuley et al.).
- **Kích thước**: 2,023,070 ratings · 1,210,271 users · 249,274 products.
- **Trường dữ liệu gốc**:
  - `UserId`, `ProductId`, `Rating (1-5)`, `Timestamp`.
- **Feature đã tạo (Notebook 01)**:
  - **User**: `num_ratings`, `avg_rating`, `positive_ratio`.
  - **Product**: `num_ratings`, `avg_rating`, `popularity_score`.
- **Đặc điểm**: Phân bố rating lệch phải (mean 4.15, median 5.0); dữ liệu cực kỳ thưa (sparsity ~99.99%).

---

## 5. Method

### 5.1 Quy trình xử lý dữ liệu

1. **EDA (Notebook 01)**: Xác nhận chất lượng, tạo feature hành vi.
2. **Preprocessing (Notebook 02)**:
   - Kiểm tra completeness, validity, uniqueness.
   - Outlier analysis (user hoạt động bất thường, item quá phổ biến).
   - Không dùng Min-Max normalization để giữ thang điểm 1–5.
   - Lưu `ratings_Beauty_processed_clean.npz`.
3. **Modeling (Notebook 03)**:
   - Load raw CSV + feature `.npz`.
   - `GroupShuffleSplit` tránh user leakage.
   - Train Pure NumPy MF với feature enrichment.

### 5.2 Phương pháp: Collaborative Filtering Matrix Factorization 

**Mô hình dự đoán:**\
$$\hat{r}_{ui} = \mu + b_u + b_i+\mathbf{p}_u^\top \mathbf{q}_i+\mathbf{f}_u^\top \mathbf{w}_u+ \mathbf{f}_i^\top \mathbf{w}_i$$

**Loss function:**\
$$\mathcal{L}= \sum_{(u,i)} (r_{ui} - \hat{r}_{ui})^2+ \lambda \left(\|\mathbf{p}_u\|_2^2 +\|\mathbf{q}_i\|_2^2 +\|\mathbf{w}_u\|_2^2 +\|\mathbf{w}_i\|_2^2\right)$$

**Trong đó:**
| Ký hiệu | Ý nghĩa |
|--------|---------|
| **$\( r_{ui} \)$** | Rating thật mà user \(u\) dành cho item \(i\). |
| **$\( \hat{r}_{ui} \)$** | Rating dự đoán của mô hình. |
| **$\( \mu \)$** | Global mean – trung bình rating của toàn bộ dataset. |
| **$\( b_u \)$** | User bias – xu hướng người dùng chấm cao/thấp hơn mức trung bình. |
| **$\( b_i \)$** | Item bias – “độ dễ được chấm cao/thấp” của item. |
| **$\( \mathbf{p}_u \)$** | User latent vector (k-chiều) – biểu diễn sở thích ẩn của user. |
| **$\( \mathbf{q}_i \)$** | Item latent vector (k-chiều) – mô tả đặc tính tiềm ẩn của item. |
| **$\( \mathbf{f}_u \)$** | User feature vector – đặc trưng của user (nếu có). |
| **$\( \mathbf{w}_u \)$** | Trọng số cho user features. |
| **$\( \mathbf{f}_i \)$** | Item feature vector – đặc trưng của item (nếu có). |
| **$\( \mathbf{w}_i \)$** | Trọng số cho item features. |
| **$\( \lambda \)$** | Hệ số regularization – giảm overfitting bằng cách phạt norm của các vector. |
| **$\( \|\cdot\|_2^2 \)$** | L2 norm squared – hình phạt độ lớn tham số, giúp mô hình ổn định. |

### 5.3 Triển khai thuần NumPy
Một số kỹ thuật chính:

- ID encoding bằng `np.unique` -> ánh xạ `UserId/ProductId` sang chỉ số.
- Latent factors & bias lưu dạng `np.float64` để tránh overflow.
- SGD vectorized:
  - `np.add.at` để accumulate gradient.
  - Glorot initialization cho `user_factors`, `item_factors`.
  - Gradient clipping + learning-rate scheduling (`lr=0.001` default).
- Feature terms (`user_feat_matrix @ user_feat_weights`).

---

## 6. Installation & Setup

```bash
git clone https://github.com/gracebml/amazon-ratings.git
cd amazon-ratings
python -m venv .venv
.venv\\Scripts\\activate         # hoặc source .venv/bin/activate trên Linux/Mac
pip install -r requirements.txt
```

Yêu cầu thêm:
- Python 3.10+
- JupyterLab hoặc VS Code với Jupyter extension
- ~5 GB disk trống (dataset + intermediate NPZ)

---

## 7. Usage

1. **Notebook 01 – Data Exploration & Feature Engineering**
   - `jupyter lab notebooks/01_data_exploration.ipynb`
   - Chạy toàn bộ để tạo `user_features.npz`, `product_features.npz`.
2. **Notebook 02 – Preprocessing**
   - `jupyter lab notebooks/02_preprocessing.ipynb`
   - Thực hiện validation, outlier checks, lưu `ratings_Beauty_processed_clean.npz`.
3. **Notebook 03 – Modeling & Evaluation**
   - `jupyter lab notebooks/03_modeling.ipynb`
   - Train Pure NumPy MF, đo RMSE/MAE, xem top-N recommendation.

---

## 8. Results

| Split | RMSE | MAE | Ghi chú |
|-------|------|-----|---------|
| Train | **1.13** | **0.6** | Mô hình fit tốt dữ liệu lịch sử |
| Test  | **1.314** | **1.052** | Có overfitting nhẹ nhưng chấp nhận được với dữ liệu thưa |

- **Trực quan hóa**: Notebook 01 cung cấp histogram rating, phân bố hoạt động user/item, heatmap tương quan feature.
- **Phân tích**:
  - Feature behavior giúp giảm cold-start so với baseline MF thuần.
  - Overfitting đến từ sparsity cao và việc thiếu signal tiêu cực -> cần regularization mạnh hơn / implicit feedback.
    
---

## 9. Project Structure

```
amazon-ratings/
├── data/
│   ├── raw/ratings_Beauty.csv          # Nguồn dữ liệu gốc
│   └── processed/*.npz                 # Feature & dataset sau preprocessing
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA + feature engineering
│   ├── 02_preprocessing.ipynb          # Validation + outlier check + save clean NPZ
│   └── 03_modeling.ipynb               # Pure NumPy MF training/evaluation
├── src/
│   ├── data_preprocessing.py           # Hàm load/validate/save dữ liệu
|   ├── visualization.py                # Visualize các phân tích từ trả lời cho các câu hỏi
│   └── models.py                       # PureNumpyMF + utilities (GroupSplit, metrics)
├── requirements.txt
└── README.md
```

---

## 10. Challenges & Solutions

| Challenge | Giải pháp |
|-----------|-----------|
| Stratified `train_test_split` thất bại vì user chỉ có 1 rating | Chuyển sang `GroupShuffleSplit` để giữ nguyên user trong 1 tập |
| Gradient overflow & `RuntimeWarning` khi train MF | Dùng `np.float64`, Glorot init, gradient clipping, giảm learning rate |
| Metric HR@k evaluation quá chậm với 2M ratings | Tạm bỏ HR@k, tập trung RMSE/MAE; chuẩn bị sampling cho tương lai |
| Yêu cầu thuần NumPy (không pandas) | Viết toàn bộ data loader, feature merger, SGD bằng NumPy vectorized operations |

---

## 11. Future Improvements

1. **Implicit Feedback**: Thêm view/cart data và Bayesian Personalized Ranking.
2. **Regularization động**: Adaptive λ dựa trên số lượt rating của user/item.
3. **Temporal Dynamics**: Thêm decay trên timestamp để bắt trend.
4. **Diversity & Novelty**: Penalize các sản phẩm quá phổ biến, áp dụng re-ranking.
5. **API / Service**: Đóng gói model thành REST/gRPC để dễ deploy.

---

## 12. Contributors

- **Bang My Linh** – Data Science student & project owner.
  - **Contact**: mylinhbangus@gmail.com

---

## 13. License

- **Code**: MIT License
- **Dataset**: Amazon Product Data – chỉ dùng cho mục đích nghiên cứu phi thương mại. Tuân thủ điều khoản của Julian McAuley & Amazon.

