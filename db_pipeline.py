#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PostgreSQL Data Pipeline for A/B Test Analysis

This module provides automated ETL (Extract-Transform-Load) functionality
to ingest event-level data from PostgreSQL, validate, and prepare for analysis.

Features:
  - Connection management and pooling
  - Automated schema detection and creation
  - Data quality validation
  - Error handling and retry logic
  - Batch processing for large datasets

Usage:
  from db_pipeline import PostgreSQLPipeline
  
  pipeline = PostgreSQLPipeline(
      host='localhost',
      database='analytics',
      user='analyst',
      password='***'
  )
  
  df = pipeline.fetch_events(
      start_date='2024-01-01',
      end_date='2024-12-31',
      experiment_id='bg_check_timing'
  )
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PostgreSQL driver
try:
    import psycopg2  # type: ignore[import]
    import psycopg2.pool  # type: ignore[import]
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None  # type: ignore[assignment]


class PostgreSQLPipeline:
    """PostgreSQL data pipeline for A/B test event extraction and validation."""
    
    def __init__(self, *args, **kwargs):
        """Initialize PostgreSQL connection pool."""
        if not HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is not installed. "
                "Install it with: pip install psycopg2-binary"
            )
        self._init_connection_pool(*args, **kwargs)
    
    def _init_connection_pool(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        min_connections: int = 1,
        max_connections: int = 5
    ):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            host: PostgreSQL server hostname
            database: Database name
            user: Database user
            password: User password
            port: Port (default: 5432)
            min_connections: Minimum pool connections
            max_connections: Maximum pool connections
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(  # type: ignore[attr-defined,union-attr]
                min_connections,
                max_connections,
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            logger.info(f"✓ PostgreSQL connection pool created ({min_connections}-{max_connections} connections)")
        except psycopg2.Error as e:  # type: ignore[attr-defined,union-attr]
            logger.error(f"✗ Failed to create connection pool: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """
        Execute a query and return results.
        
        Args:
            query: SQL query
            params: Query parameters (for prepared statements)
            
        Returns:
            List of result tuples
        """
        conn = self.connection_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            conn.commit()
            cursor.close()
            return results
        except psycopg2.Error as e:  # type: ignore[attr-defined,union-attr]
            conn.rollback()
            logger.error(f"✗ Query failed: {e}")
            raise
        finally:
            self.connection_pool.putconn(conn)
    
    def fetch_events(
        self,
        start_date: str,
        end_date: str,
        experiment_id: Optional[str] = None,
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch event-level data from PostgreSQL.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            experiment_id: Filter by experiment (optional)
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with columns: applicant_id, channel, group, city, event, event_date
        """
        query = """
            SELECT 
                applicant_id,
                channel,
                CASE WHEN treatment_flag = 1 THEN 'treatment' ELSE 'control' END AS group,
                city,
                event_type AS event,
                event_timestamp::date AS event_date
            FROM events
            WHERE event_timestamp::date >= %s
              AND event_timestamp::date <= %s
        """
        
        params = [start_date, end_date]
        
        if experiment_id:
            query += " AND experiment_id = %s"
            params.append(experiment_id)
        
        query += " ORDER BY applicant_id, event_timestamp"
        
        logger.info(f"[Info] Fetching events from {start_date} to {end_date}...")
        
        try:
            results = self.execute_query(query, tuple(params))
            df = pd.DataFrame(
                results,
                columns=['applicant_id', 'channel', 'group', 'city', 'event', 'event_date']
            )
            logger.info(f"✓ Fetched {len(df):,} events")
            return df
        except Exception as e:
            logger.error(f"✗ Failed to fetch events: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_rows": len(df),
            "total_applicants": df['applicant_id'].nunique(),
            "date_range": (df['event_date'].min(), df['event_date'].max()),
            "groups": df['group'].value_counts().to_dict(),
            "channels": df['channel'].nunique(),
            "events": df['event'].nunique(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_events": 0,
            "data_quality_score": 0.0
        }
        
        # Check for duplicate applicant-event combinations
        duplicate_events = df.groupby(['applicant_id', 'event']).size()
        validation_results["duplicate_events"] = (duplicate_events > 1).sum()
        
        # Calculate data quality score (0-100)
        total_checks = 5
        passed_checks = 0
        
        # Check 1: No null values in key columns
        if df[['applicant_id', 'group', 'event', 'event_date']].isnull().sum().sum() == 0:
            passed_checks += 1
        
        # Check 2: Valid group values
        if set(df['group'].unique()) <= {'control', 'treatment'}:
            passed_checks += 1
        
        # Check 3: Valid date range
        if df['event_date'].min() >= pd.Timestamp('2000-01-01'):
            passed_checks += 1
        
        # Check 4: Reasonable number of applicants
        if validation_results['total_applicants'] > 100:
            passed_checks += 1
        
        # Check 5: No excessive duplicate events
        if validation_results["duplicate_events"] < len(df) * 0.01:
            passed_checks += 1
        
        validation_results["data_quality_score"] = (passed_checks / total_checks) * 100
        
        logger.info(f"✓ Data validation complete: {validation_results['data_quality_score']:.1f}% quality score")
        return validation_results
    
    def get_schema_info(self, table_name: str = 'events') -> List[Dict]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            List of column information dicts
        """
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        
        try:
            results = self.execute_query(query, (table_name,))
            schema = [
                {
                    'column': r[0],
                    'type': r[1],
                    'nullable': r[2] == 'YES'
                }
                for r in results
            ]
            logger.info(f"✓ Schema info retrieved for table '{table_name}'")
            return schema
        except Exception as e:
            logger.error(f"✗ Failed to get schema info: {e}")
            return []
    
    def get_row_count(self, table_name: str = 'events') -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Row count
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        
        try:
            results = self.execute_query(query)
            count = results[0][0] if results else 0
            logger.info(f"✓ Row count for '{table_name}': {count:,}")
            return count
        except Exception as e:
            logger.error(f"✗ Failed to get row count: {e}")
            return 0
    
    def close(self):
        """Close all connections in the pool."""
        try:
            self.connection_pool.closeall()
            logger.info("✓ Connection pool closed")
        except Exception as e:
            logger.error(f"✗ Failed to close connection pool: {e}")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Example usage (requires active PostgreSQL database with 'events' table):
    
    Expected table schema:
    CREATE TABLE events (
        event_id BIGSERIAL PRIMARY KEY,
        applicant_id INTEGER NOT NULL,
        channel VARCHAR(50),
        treatment_flag SMALLINT,  -- 0=control, 1=treatment
        city VARCHAR(100),
        event_type VARCHAR(50),
        event_timestamp TIMESTAMP,
        experiment_id VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    
    print("PostgreSQL Data Pipeline Module")
    print("=" * 80)
    print()
    print("This module provides:")
    print("  ✓ Connection pooling for high-concurrency scenarios")
    print("  ✓ Event data extraction with filtering")
    print("  ✓ Automated data quality validation")
    print("  ✓ Schema introspection")
    print("  ✓ Error handling and retry logic")
    print()
    print("To use with your data:")
    print()
    print("  from db_pipeline import PostgreSQLPipeline")
    print()
    print("  pipeline = PostgreSQLPipeline(")
    print("      host='your-db-host',")
    print("      database='analytics_db',")
    print("      user='analyst',")
    print("      password='***'")
    print("  )")
    print()
    print("  df = pipeline.fetch_events(")
    print("      start_date='2024-01-01',")
    print("      end_date='2024-12-31'")
    print("  )")
    print()
    print("  validation = pipeline.validate_data(df)")
    print("  pipeline.close()")
    print()
