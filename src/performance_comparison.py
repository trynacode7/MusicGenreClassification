"""
Performance comparison between real-time and batch prediction modes.

This script compares the performance characteristics of real-time vs batch
prediction for music genre classification.
"""

import numpy as np
import pandas as pd
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from realtime_prediction import RealTimePredictor

class PerformanceComparator:
    """Compare real-time vs batch prediction performance."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the performance comparator."""
        self.predictor = RealTimePredictor(models_dir)
        self.results = {
            'real_time': [],
            'batch': [],
            'model_comparison': {}
        }
    
    def test_real_time_performance(self, test_files: List[str], num_runs: int = 5) -> Dict:
        """
        Test real-time prediction performance.
        
        Args:
            test_files: List of audio file paths to test
            num_runs: Number of runs per file for averaging
            
        Returns:
            Performance metrics
        """
        print("Testing real-time prediction performance...")
        
        real_time_results = {
            'latencies': [],
            'accuracies': [],
            'throughput': [],
            'model_latencies': {'random_forest': [], 'svm': [], 'cnn': []}
        }
        
        for file_path in test_files:
            if not os.path.exists(file_path):
                continue
                
            print(f"Testing: {os.path.basename(file_path)}")
            
            # Load audio
            import librosa
            audio_data, sr = librosa.load(file_path, sr=22050)
            
            for run in range(num_runs):
                start_time = time.time()
                
                # Real-time prediction
                predictions = self.predictor.compare_models(audio_data)
                
                total_latency = time.time() - start_time
                real_time_results['latencies'].append(total_latency)
                
                # Record individual model latencies
                for model_name, result in predictions.items():
                    if 'latency' in result:
                        real_time_results['model_latencies'][model_name].append(result['latency'])
                
                # Calculate throughput (predictions per second)
                throughput = 1.0 / total_latency if total_latency > 0 else 0
                real_time_results['throughput'].append(throughput)
        
        return real_time_results
    
    def test_batch_performance(self, test_files: List[str], batch_sizes: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Test batch prediction performance.
        
        Args:
            test_files: List of audio file paths to test
            batch_sizes: Different batch sizes to test
            
        Returns:
            Performance metrics
        """
        print("Testing batch prediction performance...")
        
        batch_results = {
            'batch_sizes': batch_sizes,
            'latencies': [],
            'throughput': [],
            'efficiency': []
        }
        
        import librosa
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Process files in batches
            batch_latencies = []
            
            for i in range(0, len(test_files), batch_size):
                batch_files = test_files[i:i+batch_size]
                
                start_time = time.time()
                
                # Process batch
                for file_path in batch_files:
                    if os.path.exists(file_path):
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        # Simulate batch processing (no individual predictions)
                        _ = self.predictor.extract_features(audio_data)
                
                batch_latency = time.time() - start_time
                batch_latencies.append(batch_latency)
            
            # Calculate metrics
            avg_latency = np.mean(batch_latencies)
            total_files = len([f for f in test_files if os.path.exists(f)])
            throughput = total_files / avg_latency if avg_latency > 0 else 0
            efficiency = throughput / batch_size  # Files per second per batch item
            
            batch_results['latencies'].append(avg_latency)
            batch_results['throughput'].append(throughput)
            batch_results['efficiency'].append(efficiency)
        
        return batch_results
    
    def compare_model_performance(self, test_files: List[str]) -> Dict:
        """
        Compare performance across different models.
        
        Args:
            test_files: List of audio file paths to test
            
        Returns:
            Model comparison metrics
        """
        print("Comparing model performance...")
        
        model_results = {
            'random_forest': {'latencies': [], 'accuracies': []},
            'svm': {'latencies': [], 'accuracies': []},
            'cnn': {'latencies': [], 'accuracies': []}
        }
        
        import librosa
        
        for file_path in test_files[:10]:  # Test with first 10 files
            if not os.path.exists(file_path):
                continue
                
            audio_data, sr = librosa.load(file_path, sr=22050)
            
            # Test each model
            for model_name in ['random_forest', 'svm', 'cnn']:
                if self.predictor.models[model_name] is not None:
                    result = self.predictor.predict_genre(audio_data, model_name)
                    
                    if 'latency' in result:
                        model_results[model_name]['latencies'].append(result['latency'])
                    
                    if 'confidence' in result:
                        model_results[model_name]['accuracies'].append(result['confidence'])
        
        return model_results
    
    def generate_performance_report(self, test_files: List[str], output_dir: str = "results"):
        """
        Generate comprehensive performance report.
        
        Args:
            test_files: List of audio file paths to test
            output_dir: Directory to save results
        """
        print("Generating performance report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run performance tests
        real_time_results = self.test_real_time_performance(test_files)
        batch_results = self.test_batch_performance(test_files)
        model_results = self.compare_model_performance(test_files)
        
        # Create visualizations
        self._create_latency_comparison_plot(real_time_results, batch_results, output_dir)
        self._create_model_comparison_plot(model_results, output_dir)
        self._create_throughput_plot(batch_results, output_dir)
        
        # Save results to CSV
        self._save_results_to_csv(real_time_results, batch_results, model_results, output_dir)
        
        # Print summary
        self._print_performance_summary(real_time_results, batch_results, model_results)
    
    def _create_latency_comparison_plot(self, real_time_results: Dict, batch_results: Dict, output_dir: str):
        """Create latency comparison visualization."""
        plt.figure(figsize=(12, 8))
        
        # Real-time latency distribution
        plt.subplot(2, 2, 1)
        plt.hist(real_time_results['latencies'], bins=20, alpha=0.7, color='blue', label='Real-time')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.title('Real-time Prediction Latency Distribution')
        plt.legend()
        
        # Batch latency by batch size
        plt.subplot(2, 2, 2)
        plt.plot(batch_results['batch_sizes'], batch_results['latencies'], 'ro-', label='Batch')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Latency (seconds)')
        plt.title('Batch Prediction Latency vs Batch Size')
        plt.legend()
        
        # Model-specific latencies
        plt.subplot(2, 2, 3)
        model_latencies = real_time_results['model_latencies']
        models = list(model_latencies.keys())
        latencies = [np.mean(model_latencies[model]) for model in models if model_latencies[model]]
        plt.bar(models, latencies, color=['green', 'orange', 'red'])
        plt.xlabel('Model')
        plt.ylabel('Average Latency (seconds)')
        plt.title('Model-specific Latency Comparison')
        plt.xticks(rotation=45)
        
        # Throughput comparison
        plt.subplot(2, 2, 4)
        real_time_throughput = np.mean(real_time_results['throughput'])
        batch_throughput = np.max(batch_results['throughput'])
        
        categories = ['Real-time', 'Batch (Optimal)']
        throughputs = [real_time_throughput, batch_throughput]
        plt.bar(categories, throughputs, color=['blue', 'green'])
        plt.ylabel('Throughput (predictions/second)')
        plt.title('Throughput Comparison')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_comparison_plot(self, model_results: Dict, output_dir: str):
        """Create model comparison visualization."""
        plt.figure(figsize=(15, 5))
        
        # Latency comparison
        plt.subplot(1, 3, 1)
        models = []
        latencies = []
        for model, results in model_results.items():
            if results['latencies']:
                models.append(model.replace('_', ' ').title())
                latencies.append(np.mean(results['latencies']))
        
        plt.bar(models, latencies, color=['green', 'orange', 'red'])
        plt.xlabel('Model')
        plt.ylabel('Average Latency (seconds)')
        plt.title('Model Latency Comparison')
        plt.xticks(rotation=45)
        
        # Confidence comparison
        plt.subplot(1, 3, 2)
        confidences = []
        for model, results in model_results.items():
            if results['accuracies']:
                confidences.append(np.mean(results['accuracies']))
        
        plt.bar(models, confidences, color=['green', 'orange', 'red'])
        plt.xlabel('Model')
        plt.ylabel('Average Confidence')
        plt.title('Model Confidence Comparison')
        plt.xticks(rotation=45)
        
        # Latency vs Confidence scatter
        plt.subplot(1, 3, 3)
        for i, model in enumerate(models):
            if i < len(latencies) and i < len(confidences):
                plt.scatter(latencies[i], confidences[i], s=100, label=model)
        
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Confidence')
        plt.title('Latency vs Confidence Trade-off')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_throughput_plot(self, batch_results: Dict, output_dir: str):
        """Create throughput analysis plot."""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(batch_results['batch_sizes'], batch_results['throughput'], 'bo-', label='Total Throughput')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (files/second)')
        plt.title('Batch Processing Throughput')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(batch_results['batch_sizes'], batch_results['efficiency'], 'ro-', label='Efficiency')
        plt.xlabel('Batch Size')
        plt.ylabel('Efficiency (files/second/batch_item)')
        plt.title('Batch Processing Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results_to_csv(self, real_time_results: Dict, batch_results: Dict, model_results: Dict, output_dir: str):
        """Save results to CSV files."""
        # Real-time results
        rt_df = pd.DataFrame({
            'latency': real_time_results['latencies'],
            'throughput': real_time_results['throughput']
        })
        rt_df.to_csv(f"{output_dir}/real_time_performance.csv", index=False)
        
        # Batch results
        batch_df = pd.DataFrame({
            'batch_size': batch_results['batch_sizes'],
            'latency': batch_results['latencies'],
            'throughput': batch_results['throughput'],
            'efficiency': batch_results['efficiency']
        })
        batch_df.to_csv(f"{output_dir}/batch_performance.csv", index=False)
        
        # Model comparison
        model_data = []
        for model, results in model_results.items():
            if results['latencies'] and results['accuracies']:
                model_data.append({
                    'model': model,
                    'avg_latency': np.mean(results['latencies']),
                    'std_latency': np.std(results['latencies']),
                    'avg_confidence': np.mean(results['accuracies']),
                    'std_confidence': np.std(results['accuracies'])
                })
        
        model_df = pd.DataFrame(model_data)
        model_df.to_csv(f"{output_dir}/model_performance.csv", index=False)
    
    def _print_performance_summary(self, real_time_results: Dict, batch_results: Dict, model_results: Dict):
        """Print performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        # Real-time performance
        rt_avg_latency = np.mean(real_time_results['latencies'])
        rt_avg_throughput = np.mean(real_time_results['throughput'])
        
        print(f"\nReal-time Prediction:")
        print(f"  Average Latency: {rt_avg_latency:.3f} seconds")
        print(f"  Average Throughput: {rt_avg_throughput:.2f} predictions/second")
        
        # Batch performance
        optimal_batch_idx = np.argmax(batch_results['throughput'])
        optimal_batch_size = batch_results['batch_sizes'][optimal_batch_idx]
        optimal_throughput = batch_results['throughput'][optimal_batch_idx]
        
        print(f"\nBatch Prediction (Optimal):")
        print(f"  Optimal Batch Size: {optimal_batch_size}")
        print(f"  Maximum Throughput: {optimal_throughput:.2f} files/second")
        
        # Model comparison
        print(f"\nModel Performance:")
        for model, results in model_results.items():
            if results['latencies'] and results['accuracies']:
                avg_latency = np.mean(results['latencies'])
                avg_confidence = np.mean(results['accuracies'])
                print(f"  {model.replace('_', ' ').title()}: "
                      f"{avg_latency:.3f}s latency, {avg_confidence:.3f} confidence")
        
        # Recommendations
        print(f"\nRecommendations:")
        if rt_avg_latency < 1.0:
            print("  ✓ Real-time prediction is suitable for interactive applications")
        else:
            print("  ⚠ Real-time prediction may be too slow for real-time applications")
        
        if optimal_throughput > rt_avg_throughput * 2:
            print("  ✓ Batch processing provides significant throughput improvement")
        else:
            print("  ⚠ Batch processing provides modest throughput improvement")
        
        print("="*60)


def main():
    """Main function for performance comparison."""
    print("=== Music Genre Classification Performance Comparison ===\n")
    
    # Find test files
    test_files = []
    genres_dir = Path("Data/genres_original")
    
    if genres_dir.exists():
        for genre_dir in genres_dir.iterdir():
            if genre_dir.is_dir():
                audio_files = list(genre_dir.glob("*.wav"))[:5]  # Take first 5 files per genre
                for audio_file in audio_files:
                    test_files.append(str(audio_file))
    
    if not test_files:
        print("No test files found. Please ensure Data/genres_original/ exists.")
        return
    
    print(f"Found {len(test_files)} test files")
    
    # Initialize comparator
    comparator = PerformanceComparator()
    
    # Generate performance report
    comparator.generate_performance_report(test_files)
    
    print("\nPerformance comparison completed!")
    print("Results saved to 'results/' directory")


if __name__ == "__main__":
    main()
