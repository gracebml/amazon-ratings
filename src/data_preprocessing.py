import numpy as np
from typing import Tuple, Dict, Any
import os

def load_data(file_path: str) -> np.ndarray:
    
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=',',
            skip_header=1,
            dtype=[
                ('UserId', 'U50'),
                ('ProductId', 'U50'),
                ('Rating', 'f4'),
                ('Timestamp', 'i8')
            ],
            encoding='utf-8'
        )
        print(f" Đọc dữ liệu thành công từ {file_path}")
        print(f"  Số dòng: {len(data):,}")
        return data
    except FileNotFoundError:
        print(f" Lỗi: Không tìm thấy file '{file_path}'")
        raise
    except Exception as e:
        print(f" Lỗi khi đọc file: {str(e)}")
        raise


def check_missing_values(data: np.ndarray) -> Tuple[int, Dict[str, int]]:

    print("\n=== KIỂM TRA MISSING VALUES ===")
    missing_counts = {}
    
    # Vectorized check cho tất cả columns
    for col_name in data.dtype.names:
        col_data = data[col_name]
        
        # Sử dụng np.where và broadcasting thay vì if-else
        is_numeric = np.issubdtype(col_data.dtype, np.number)
        
        # Universal function áp dụng cho toàn bộ array
        count = np.sum(np.isnan(col_data)) if is_numeric else \
                np.sum((col_data == '') | (col_data == b''))
        
        missing_counts[col_name] = int(count)
        status = " THIẾU" if count > 0 else " Đầy đủ"
        print(f"  {col_name:15s}: {status:12s} ({count:,} missing)")
    
    # Broadcasting sum
    total_missing = np.sum(list(missing_counts.values()))
    print(f"\n  Tổng missing values: {total_missing:,}")
    return int(total_missing), missing_counts

def validate_data(data: np.ndarray) -> Tuple[np.ndarray, int]:
    print("\n=== KIỂM TRA TÍNH HỢP LỆ ===")
    initial_count = len(data)
    
    # Vectorized boolean operations với broadcasting
    valid_mask = (
        (data['Rating'] >= 1) & 
        (data['Rating'] <= 5) & 
        (data['Timestamp'] > 0) &
        (data['UserId'] != '') &
        (data['ProductId'] != '')
    )
    
    # Fancy indexing để lọc
    clean_data = data[valid_mask]
    removed_count = initial_count - len(clean_data)
    
    print(f"  Số dòng ban đầu: {initial_count:,}")
    print(f"  Số dòng không hợp lệ: {removed_count:,}")
    print(f"  Số dòng còn lại: {len(clean_data):,}")
    print(f"  Tỷ lệ giữ lại: {len(clean_data)/initial_count*100:.2f}%")
    
    return clean_data, removed_count


def run_validation_checks(data: np.ndarray) -> Dict[str, Any]:
    """
    Execute and print all validation checks used in notebook 02.
    
    Returns a dictionary with aggregated statistics for further use.
    """
    print("=" * 70)
    print("BƯỚC 2: KIỂM TRA TÍNH HỢP LỆ DỮ LIỆU")
    print("=" * 70)

    stats: Dict[str, Any] = {}

    total_missing, missing_counts = check_missing_values(data)
    stats["missing"] = {"total": total_missing, "per_column": missing_counts}

    ratings = data["Rating"]
    invalid_ratings = int(np.sum((ratings < 1) | (ratings > 5)))
    unique_ratings, rating_counts = np.unique(ratings, return_counts=True)

    print("\n[2.2] KIỂM TRA KHOẢNG GIÁ TRỊ HỢP LỆ")
    print("-" * 70)
    print(f"Rating ngoài [1,5]: {invalid_ratings:,}")
    for val, cnt in zip(unique_ratings, rating_counts):
        percent = cnt / len(ratings) * 100
        print(f"  Rating {val:.1f}: {cnt:,} ({percent:.1f}%)")
    if invalid_ratings == 0:
        print("\n PASS: Tất cả ratings hợp lệ [1-5]")
    stats["rating_distribution"] = {
        "values": unique_ratings.tolist(),
        "counts": rating_counts.tolist(),
        "invalid": invalid_ratings,
    }

    print("\n[2.3] KIỂM TRA TRÙNG LẶP")
    print("-" * 70)
    users = data["UserId"]
    items = data["ProductId"]
    unique_users = np.unique(users).size
    unique_items = np.unique(items).size
    sparsity = 1 - len(data) / (unique_users * unique_items)
    print(f"Users duy nhất: {unique_users:,}")
    print(f"Products duy nhất: {unique_items:,}")
    print(f"Sparsity: {sparsity*100:.2f}%")

    user_product_pairs = data[["UserId", "ProductId"]]
    unique_pairs = np.unique(user_product_pairs).size
    duplicate_pairs = len(data) - unique_pairs
    dup_percent = duplicate_pairs / len(data) * 100
    print(f"\nCặp (user,product) trùng: {duplicate_pairs:,} ({dup_percent:.2f}%)")
    if duplicate_pairs > 0:
        print(" Giữ nguyên: temporal reviews hợp lệ")

    stats["uniqueness"] = {
        "unique_users": unique_users,
        "unique_items": unique_items,
        "sparsity": sparsity,
        "duplicate_pairs": duplicate_pairs,
    }
    return stats


def run_outlier_checks(data: np.ndarray) -> Dict[str, Any]:
    print("=" * 70)
    print("BƯỚC 3: PHÁT HIỆN GIÁ TRỊ BẤT THƯỜNG")
    print("=" * 70)

    report: Dict[str, Any] = {}

    def _analyze_counts(values: np.ndarray, label: str) -> Dict[str, Any]:
        _, counts = np.unique(values, return_counts=True)
        q1, q3 = np.percentile(counts, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outliers = int(np.sum(counts > upper_bound))
        stats = {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "iqr_threshold": float(upper_bound),
            "outliers": outliers,
            "outlier_percent": outliers / counts.size * 100,
        }
        print(f"\n[{label}]")
        print("-" * 70)
        print(f"Ratings/{label.lower()} - Min: {stats['min']:,}, Max: {stats['max']:,}")
        print(f"Ratings/{label.lower()} - Mean: {stats['mean']:.1f}, Median: {stats['median']:.0f}")
        print(f"\nOutlier threshold (IQR): {stats['iqr_threshold']:.0f}")
        print(f"{label} vượt ngưỡng: {stats['outliers']:,} ({stats['outlier_percent']:.1f}%)")
        return stats

    report["user_behavior"] = _analyze_counts(data["UserId"], "USER BEHAVIOR OUTLIERS")
    print("\n Giữ nguyên: Power users có giá trị cho CF")
    report["product_popularity"] = _analyze_counts(data["ProductId"], "PRODUCT POPULARITY OUTLIERS")
    print("\n Giữ nguyên: Popular products có giá trị cho recommendation")
    return report

def save_processed_data(data: np.ndarray, output_path: str):
    print(f"\n=== LƯU DỮ LIỆU ===")
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Created directory: {output_dir}")
    
    # Save as .npz only 
    full_path = f"{output_path}.npz"
    
    # Remove old .npy file if exists
    old_npy_path = f"{output_path}.npy"
    if os.path.exists(old_npy_path):
        try:
            os.remove(old_npy_path)
            print(f"  Removed old .npy file: {old_npy_path}")
        except Exception as e:
            print(f"  Warning: Could not remove old .npy file: {e}")
    
    # Remove old .npz file if exists
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
        except Exception as e:
            print(f"  Warning: Could not remove old .npz file: {e}")
    
    try:
        # Save as compressed numpy format
        np.savez_compressed(full_path, data=data)
        print(f"  Saved: {full_path} (compressed)")
    except Exception as e:
        print(f"  Error saving file: {e}")
        raise
    
    print(f"  Kích thước: {len(data):,} records")
    
    # Verify file size
    if os.path.exists(full_path):
        file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")