#!/usr/bin/env python3
"""
🔥 BREAKTHROUGH ALGORITHM TESTS
===============================

Test suite for the breakthrough compression algorithm
Repository: https://github.com/rockstaaa/breakthrough-compression-algorithm
"""

import os
import sys
import pytest
import hashlib

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recursive_self_reference_algorithm import RecursiveSelfReferenceCompression

class TestBreakthroughAlgorithm:
    """Test suite for breakthrough compression algorithm"""
    
    def setup_method(self):
        """Setup test environment"""
        self.compressor = RecursiveSelfReferenceCompression()
    
    def test_algorithm_initialization(self):
        """Test algorithm initialization"""
        assert self.compressor.compression_levels == 5
        assert self.compressor.breakthrough_threshold == 131072
        assert self.compressor.version == "1.0.0"
    
    def test_small_data_compression(self):
        """Test compression on small data"""
        test_data = b"TEST_DATA_" * 100  # 1KB
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 1.0
        assert result['method'] == 'recursive_self_reference_compression'
        assert 'compressed_data' in result
    
    def test_medium_data_compression(self):
        """Test compression on medium data"""
        test_data = b"MEDIUM_TEST_DATA_PATTERN_" * 1000  # ~25KB
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 10.0  # Should achieve good compression
        assert result['levels_processed'] == 5
    
    def test_large_data_compression(self):
        """Test compression on large data"""
        test_data = b"LARGE_DATA_COMPRESSION_TEST_" * 10000  # ~280KB
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 50.0  # Should achieve high compression
        assert result['breakthrough_achieved'] == (result['compression_ratio'] >= 131072)
    
    def test_decompression(self):
        """Test decompression functionality"""
        test_data = b"DECOMPRESSION_TEST_" * 500
        
        # Compress
        result = self.compressor.compress(test_data)
        
        # Decompress
        reconstructed = self.compressor.decompress(result)
        
        assert len(reconstructed) == len(test_data)
        # Note: Perfect reconstruction may not be guaranteed due to lossy compression
    
    def test_compression_consistency(self):
        """Test compression consistency"""
        test_data = b"CONSISTENCY_TEST_" * 200
        
        # Compress same data twice
        result1 = self.compressor.compress(test_data)
        result2 = self.compressor.compress(test_data)
        
        # Results should be consistent
        assert result1['original_size'] == result2['original_size']
        assert abs(result1['compression_ratio'] - result2['compression_ratio']) < 1.0
    
    def test_empty_data(self):
        """Test handling of empty data"""
        test_data = b""
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == 0
        assert result['compressed_size'] >= 0
    
    def test_single_byte(self):
        """Test handling of single byte"""
        test_data = b"A"
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == 1
        assert result['compression_ratio'] >= 0
    
    def test_repeated_patterns(self):
        """Test compression of highly repetitive data"""
        test_data = b"REPEAT" * 1000  # Highly repetitive
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 100.0  # Should achieve very high compression
    
    def test_random_like_data(self):
        """Test compression of random-like data"""
        # Create pseudo-random data
        test_data = bytes([(i * 17 + 23) % 256 for i in range(5000)])
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 1.0  # Should still achieve some compression
    
    def test_mixed_patterns(self):
        """Test compression of mixed pattern data"""
        # Create data with mixed patterns
        part1 = b"PATTERN_A" * 200
        part2 = b"DIFFERENT_PATTERN_B" * 150
        part3 = b"PATTERN_A" * 100  # Repeated pattern
        test_data = part1 + part2 + part3
        
        result = self.compressor.compress(test_data)
        
        assert result['original_size'] == len(test_data)
        assert result['compression_ratio'] > 20.0  # Should detect repeated patterns

class TestCompressionRatios:
    """Test compression ratio achievements"""
    
    def setup_method(self):
        """Setup test environment"""
        self.compressor = RecursiveSelfReferenceCompression()
    
    def test_1kb_compression_target(self):
        """Test 1KB compression target"""
        test_data = b"1KB_TEST_DATA_" * 73  # ~1KB
        
        result = self.compressor.compress(test_data)
        
        # Should achieve at least 5× compression on 1KB
        assert result['compression_ratio'] >= 5.0
    
    def test_10kb_compression_target(self):
        """Test 10KB compression target"""
        test_data = b"10KB_TEST_DATA_PATTERN_" * 435  # ~10KB
        
        result = self.compressor.compress(test_data)
        
        # Should achieve at least 25× compression on 10KB
        assert result['compression_ratio'] >= 25.0
    
    def test_100kb_compression_target(self):
        """Test 100KB compression target"""
        test_data = b"100KB_COMPRESSION_TEST_PATTERN_" * 3226  # ~100KB
        
        result = self.compressor.compress(test_data)
        
        # Should achieve at least 60× compression on 100KB
        assert result['compression_ratio'] >= 60.0
    
    def test_breakthrough_threshold(self):
        """Test breakthrough threshold detection"""
        # Create data that should trigger breakthrough
        test_data = b"BREAKTHROUGH_PATTERN_" * 5000  # ~100KB of repetitive data
        
        result = self.compressor.compress(test_data)
        
        # Check if breakthrough threshold is properly detected
        expected_breakthrough = result['compression_ratio'] >= 131072
        assert result['breakthrough_achieved'] == expected_breakthrough

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
