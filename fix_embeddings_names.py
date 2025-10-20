#!/usr/bin/env python3
"""
Script để tự động đổi tên thư mục embeddings sang format lowercase chuẩn.

Usage:
    python fix_embeddings_names.py
    
Hoặc để preview mà không thực sự đổi tên:
    python fix_embeddings_names.py --dry-run
"""
import os
import sys
import re
from pathlib import Path


def convert_to_lowercase_username(name: str) -> str:
    """
    Convert tên thư mục sang format lowercase chuẩn.
    
    Examples:
        102220347 -> le_quoc_viet
        Nguyen_Van_A -> nguyen_van_a
        TranThiB -> tran_thi_b (thêm underscore nếu CamelCase)
    """
    # Nếu đã là lowercase và có underscore, giữ nguyên
    if name.islower() and '_' in name:
        return name
    
    # Nếu là PascalCase hoặc camelCase, thêm underscore
    # Ví dụ: LeQuocViet -> 102220347
    name_with_underscores = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    name_with_underscores = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', name_with_underscores)
    
    # Convert to lowercase
    return name_with_underscores.lower()


def fix_embeddings_names(embeddings_dir: str = "embeddings", dry_run: bool = False):
    """
    Đổi tên các thư mục embeddings sang format chuẩn.
    
    Args:
        embeddings_dir: Đường dẫn đến thư mục embeddings
        dry_run: Nếu True, chỉ in ra mà không thực sự đổi tên
    """
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        print(f"❌ Không tìm thấy thư mục: {embeddings_dir}")
        return
    
    if not embeddings_path.is_dir():
        print(f"❌ {embeddings_dir} không phải là thư mục")
        return
    
    print(f"🔍 Scanning thư mục: {embeddings_path.absolute()}")
    print(f"{'Mode: DRY RUN (không đổi tên thực sự)' if dry_run else 'Mode: LIVE (sẽ đổi tên thực sự)'}")
    print("=" * 80)
    
    changes = []
    no_changes = []
    
    # Lấy danh sách tất cả thư mục con
    for item in embeddings_path.iterdir():
        if not item.is_dir():
            continue
        
        old_name = item.name
        new_name = convert_to_lowercase_username(old_name)
        
        if old_name == new_name:
            no_changes.append(old_name)
            print(f"✅ {old_name:<30} (đã đúng format)")
        else:
            changes.append((old_name, new_name))
            print(f"🔄 {old_name:<30} → {new_name}")
    
    print("=" * 80)
    
    # Summary
    print(f"\n📊 Tóm tắt:")
    print(f"  - Tổng số thư mục: {len(changes) + len(no_changes)}")
    print(f"  - Đã đúng format: {len(no_changes)}")
    print(f"  - Cần đổi tên: {len(changes)}")
    
    # Nếu không có gì cần đổi
    if not changes:
        print(f"\n🎉 Tất cả thư mục đã đúng format!")
        return
    
    # Xác nhận trước khi đổi tên
    if not dry_run:
        print(f"\n⚠️  Bạn có chắc chắn muốn đổi tên {len(changes)} thư mục?")
        response = input("Nhập 'yes' để xác nhận: ").strip().lower()
        
        if response != 'yes':
            print("❌ Đã hủy.")
            return
        
        print(f"\n🚀 Bắt đầu đổi tên...")
        
        # Thực hiện đổi tên
        for old_name, new_name in changes:
            old_path = embeddings_path / old_name
            new_path = embeddings_path / new_name
            
            try:
                # Kiểm tra xem tên mới đã tồn tại chưa
                if new_path.exists():
                    print(f"⚠️  SKIP: {old_name} → {new_name} (tên mới đã tồn tại!)")
                    continue
                
                old_path.rename(new_path)
                print(f"✅ DONE: {old_name} → {new_name}")
            except Exception as e:
                print(f"❌ ERROR: {old_name} → {e}")
        
        print(f"\n🎉 Hoàn thành!")
    else:
        print(f"\n💡 Để thực sự đổi tên, chạy lại script mà không có --dry-run")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tự động đổi tên thư mục embeddings sang format lowercase chuẩn"
    )
    parser.add_argument(
        "--embeddings-dir",
        default="embeddings",
        help="Đường dẫn đến thư mục embeddings (default: embeddings)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes mà không thực sự đổi tên"
    )
    
    args = parser.parse_args()
    
    fix_embeddings_names(
        embeddings_dir=args.embeddings_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

