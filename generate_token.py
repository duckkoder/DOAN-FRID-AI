#!/usr/bin/env python3
"""
Script tạo JWT token để test API
"""
import os
from datetime import datetime, timezone
import jwt

# Cấu hình JWT
JWT_SECRET = "test-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_AUDIENCE = "ai-service"
JWT_ISSUER = "test-issuer"

def create_test_token():
    """Tạo JWT token để test"""
    payload = {
        "sub": "test-user",
        "exp": int((datetime.now(timezone.utc).timestamp())) + 3600,  # 1 hour
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "aud": JWT_AUDIENCE,
        "iss": JWT_ISSUER
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

if __name__ == "__main__":
    token = create_test_token()
    print("Test JWT Token:")
    print(token)
    print("\nSử dụng token này để test API:")
    print(f"export JWT_TOKEN='{token}'")
    print("curl -H \"Authorization: Bearer $JWT_TOKEN\" http://localhost:8000/api_ai/v1/sessions/SESSION_ID/frames")
