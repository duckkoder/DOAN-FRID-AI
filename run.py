#!/usr/bin/env python3
"""
Script khởi chạy AI-Service
"""
import os
import uvicorn

if __name__ == "__main__":
    # Set default environment variables nếu chưa có
    os.environ.setdefault('JWT_SECRET', 'default-secret-key-change-in-production')
    os.environ.setdefault('JWT_AUDIENCE', 'ai-service')
    os.environ.setdefault('JWT_ISSUER', 'ai-service-issuer')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8096,
        reload=True,
        log_level="info"
    )
