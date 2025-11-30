import numpy as np
from typing import Tuple, Dict
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
        status = "✗ THIẾU" if count > 0 else "✓ Đầy đủ"
        print(f"  {col_name:15s}: {status:12s} ({count:,} missing)")
    
    # Broadcasting sum
    total_missing = np.sum(list(missing_counts.values()))
    print(f"\n  Tổng missing values: {total_missing:,}")
    return int(total_missing), missing_counts


def fill_missing_values(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:

    data_copy = data.copy()
    
    # Lấy view của Rating (memory-efficient)
    ratings = data_copy['Rating']
    
    # Boolean masking - vectorized
    missing_mask = np.isnan(ratings)
    
    if np.any(missing_mask):  # Universal function
        # Tính fill value bằng universal functions
        fill_value = np.nanmean(ratings) if strategy == 'mean' else \
                     np.nanmedian(ratings) if strategy == 'median' else \
                     float(strategy)
        
        # Fancy indexing để điền values
        ratings[missing_mask] = fill_value
        print(f" Đã điền {np.sum(missing_mask)} missing values với {strategy}={fill_value:.2f}")
    
    return data_copy


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


def detect_outliers_iqr(data: np.ndarray, column: str = 'Rating', 
                        multiplier: float = 1.5) -> np.ndarray:
    values = data[column]
    
    # Vectorized percentile calculation
    Q1, Q3 = np.percentile(values, [25, 75])
    IQR = Q3 - Q1
    
    # Broadcasting operations
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Vectorized comparison
    outliers_mask = (values < lower_bound) | (values > upper_bound)
    num_outliers = np.sum(outliers_mask)
    
    print(f"\n=== PHÁT HIỆN OUTLIERS (IQR) - {column} ===")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Số outliers: {num_outliers:,} ({num_outliers/len(data)*100:.2f}%)")
    
    return outliers_mask


def normalize_minmax(values: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    
    # Broadcasting operations
    min_val, max_val = np.min(values), np.max(values)
    
    if max_val == min_val:
        return np.full_like(values, feature_range[0], dtype=np.float32)
    
    # Vectorized computation với broadcasting
    normalized = (values - min_val) / (max_val - min_val)
    
    target_min, target_max = feature_range
    # Broadcasting scalar operations
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled.astype(np.float32)


def normalize_log(values: np.ndarray) -> np.ndarray:
    
    # Vectorized check và shift nếu cần
    min_val = np.min(values)
    shifted_values = np.where(min_val < 0, values - min_val, values)
    
    # Universal function log1p
    return np.log1p(shifted_values).astype(np.float32)


def normalize_decimal_scaling(values: np.ndarray):
    # Vectorized operations
    max_abs = np.max(np.abs(values))
    
    if max_abs == 0:
        return values.astype(np.float32)
    
    # Broadcasting computation
    d = np.ceil(np.log10(max_abs + 1))
    divisor = 10 ** d
    
    return (values / divisor).astype(np.float32)


def standardize_zscore(values: np.ndarray) -> np.ndarray:
    # Universal functions
    mean, std = np.mean(values), np.std(values)
    
    if std == 0:
        return (values - mean).astype(np.float32)
    
    # Broadcasting operations
    return ((values - mean) / std).astype(np.float32)


def apply_normalization(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    
    print(f"\n=== ÁP DỤNG CHUẨN HÓA: {method.upper()} ===")
    
    # Tạo structured array mới
    new_dtype = data.dtype.descr + [
        ('Rating_normalized', 'f4'),
        ('Timestamp_normalized', 'f4')
    ]
    
    processed_data = np.empty(len(data), dtype=new_dtype)
    
    # Copy tất cả fields cùng lúc bằng cách gán từng field (vectorized)
    # Thay vì for loop, dùng list comprehension để tạo operations vectorized
    for name in data.dtype.names:
        processed_data[name] = data[name]
    
    # Dictionary mapping methods (tránh if-elif chain)
    normalization_funcs = {
        'minmax': normalize_minmax,
        'log': normalize_log,
        'decimal': normalize_decimal_scaling,
        'zscore': standardize_zscore
    }
    
    norm_func = normalization_funcs.get(method, normalize_minmax)
    
    # Vectorized normalization
    processed_data['Rating_normalized'] = norm_func(data['Rating'].astype(np.float32))
    processed_data['Timestamp_normalized'] = norm_func(data['Timestamp'].astype(np.float32))
    
    # Broadcasting statistics
    print(f"  Rating - Original: mean={np.mean(data['Rating']):.2f}, std={np.std(data['Rating']):.2f}")
    print(f"  Rating - Normalized: mean={np.mean(processed_data['Rating_normalized']):.2f}, "
          f"std={np.std(processed_data['Rating_normalized']):.2f}")
    
    return processed_data


def save_processed_data(data: np.ndarray, output_path: str):
    """
    Save processed data as .npz (compressed) format only.
    Removes any existing .npy file with the same name.
    """
    print(f"\n=== LƯU DỮ LIỆU ===")
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Created directory: {output_dir}")
    
    # Save as .npz only (compressed format)
    full_path = f"{output_path}.npz"
    
    # Remove old .npy file if exists (to avoid confusion)
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