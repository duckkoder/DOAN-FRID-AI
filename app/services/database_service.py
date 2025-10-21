"""
Database Service - Query face embeddings from PostgreSQL pgvector
"""
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from app.core.config import settings
from app.core.logging import LoggerMixin


class DatabaseService(LoggerMixin):
    """
    Service để query embeddings từ PostgreSQL pgvector.
    Chỉ dùng cho load embeddings khi start session.
    """
    
    def __init__(self):
        super().__init__()
        self.connection = None
        self.logger.info("DatabaseService initialized", database_url=settings.DATABASE_URL)
    
    def connect(self) -> None:
        """Kết nối đến PostgreSQL database."""
        try:
            if self.connection and not self.connection.closed:
                return
            
            self.connection = psycopg2.connect(settings.DATABASE_URL)
            self.logger.info("Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error("Failed to connect to database", error=str(e))
            raise
    
    def disconnect(self) -> None:
        """Đóng kết nối database."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            self.logger.info("Disconnected from PostgreSQL database")
    
    def get_embeddings_by_student_codes(
        self,
        student_codes: List[str],
        status: str = "approved"
    ) -> List[Dict[str, Any]]:
        """
        Query embeddings cho danh sách student codes.
        Đây là query DUY NHẤT khi start session.
        
        Args:
            student_codes: List student codes (e.g., ['102220312', '102220347', ...])
            status: Status filter (default: 'approved')
        
        Returns:
            List of dicts containing:
            - student_code: str
            - student_id: int
            - embedding: list of 512 floats
            - embedding_id: int
        """
        if not student_codes:
            self.logger.warning("No student codes provided")
            return []
        
        try:
            self.connect()
            
            query = """
                SELECT 
                    id as embedding_id,
                    student_id,
                    student_code,
                    embedding
                FROM face_embeddings
                WHERE student_code = ANY(%s)
                  AND status = %s
                ORDER BY student_code, id
            """
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (student_codes, status))
                rows = cursor.fetchall()
            
            results = []
            for row in rows:
                # Convert pgvector to list
                embedding = row['embedding']
                if isinstance(embedding, str):
                    # Parse string representation "[0.1, 0.2, ...]"
                    embedding = eval(embedding)
                elif hasattr(embedding, 'tolist'):
                    # Convert numpy array to list
                    embedding = embedding.tolist()
                
                results.append({
                    'embedding_id': row['embedding_id'],
                    'student_id': row['student_id'],
                    'student_code': row['student_code'],
                    'embedding': embedding
                })
            
            self.logger.info(
                "Loaded embeddings from database",
                student_count=len(student_codes),
                embedding_count=len(results),
                avg_per_student=len(results) / len(student_codes) if student_codes else 0
            )
            
            return results
        
        except Exception as e:
            self.logger.error(
                "Failed to load embeddings from database",
                error=str(e),
                student_count=len(student_codes)
            )
            raise
    
    def get_embeddings_by_student_ids(
        self,
        student_ids: List[int],
        status: str = "approved"
    ) -> List[Dict[str, Any]]:
        """
        Query embeddings cho danh sách student IDs.
        Alternative method cho get_embeddings_by_student_codes.
        
        Args:
            student_ids: List student IDs
            status: Status filter (default: 'approved')
        
        Returns:
            List of dicts containing embedding data
        """
        if not student_ids:
            self.logger.warning("No student IDs provided")
            return []
        
        try:
            self.connect()
            
            query = """
                SELECT 
                    id as embedding_id,
                    student_id,
                    student_code,
                    embedding
                FROM face_embeddings
                WHERE student_id = ANY(%s)
                  AND status = %s
                ORDER BY student_id, id
            """
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (student_ids, status))
                rows = cursor.fetchall()
            
            results = []
            for row in rows:
                embedding = row['embedding']
                if isinstance(embedding, str):
                    embedding = eval(embedding)
                elif hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                results.append({
                    'embedding_id': row['embedding_id'],
                    'student_id': row['student_id'],
                    'student_code': row['student_code'],
                    'embedding': embedding
                })
            
            self.logger.info(
                "Loaded embeddings from database by IDs",
                student_count=len(student_ids),
                embedding_count=len(results)
            )
            
            return results
        
        except Exception as e:
            self.logger.error(
                "Failed to load embeddings by IDs",
                error=str(e),
                student_count=len(student_ids)
            )
            raise
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            self.connect()
            
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            self.logger.info("Database connection test successful")
            return True
        
        except Exception as e:
            self.logger.error("Database connection test failed", error=str(e))
            return False
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings in database."""
        try:
            self.connect()
            
            query = """
                SELECT 
                    COUNT(DISTINCT student_code) as total_students,
                    COUNT(*) as total_embeddings,
                    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_embeddings,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_embeddings,
                    COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_embeddings
                FROM face_embeddings
            """
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                stats = cursor.fetchone()
            
            return dict(stats) if stats else {}
        
        except Exception as e:
            self.logger.error("Failed to get embedding stats", error=str(e))
            return {}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


# Global singleton instance
_database_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """Get or create DatabaseService singleton."""
    global _database_service
    if _database_service is None:
        _database_service = DatabaseService()
    return _database_service
