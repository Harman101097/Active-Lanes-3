#!/usr/bin/env python3

import subprocess
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
from contextlib import redirect_stdout, redirect_stderr
from dateutil.relativedelta import relativedelta
import copy

# Import your scripts
try:
    from new_buffer import CADBufferCalculator
except ImportError:
    print("ERROR: Could not import CADBufferCalculator from new_buffer.py")
    print("Make sure 'new_buffer.py' is in the same folder as this script")
    sys.exit(1)

try:
    from volatility_tester import CADUSDVolatilityTester
except ImportError:
    print("ERROR: Could not import CADUSDVolatilityTester from volatility_tester.py")
    print("Make sure 'volatility_tester.py' is in the same folder as this script")
    sys.exit(1)

class ClientReadyBufferAnalyzer:
    def __init__(self):
        self.buffer_calculator = CADBufferCalculator()
        self.volatility_tester = CADUSDVolatilityTester()
        self.all_buffer_results = []
        self.volatility_strategies = {}
        self.final_client_results = {}
        self.original_client_results = {}
        
    def run_all_buffer_scenarios(self, verbose=False):
        """Run buffer calculations for all 32 scenarios"""
        
        scenarios = ['depreciating', 'appreciating']
        volatilities = [25, 50, 75, 90]
        time_horizons = [1, 3, 6, 12]
        
        total_combinations = len(scenarios) * len(volatilities) * len(time_horizons)
        print(f"Running {total_combinations} buffer calculations...")
        print("=" * 70)
        
        scenario_count = 0
        successful_count = 0
        
        for scenario in scenarios:
            for volatility in volatilities:
                for horizon in time_horizons:
                    scenario_count += 1
                    
                    if verbose:
                        print(f"\nBuffer Scenario {scenario_count}/{total_combinations}:")
                        print(f"  {scenario.upper()}, {volatility}th percentile, {horizon} month(s)")
                    else:
                        print(f"Buffer {scenario_count}/{total_combinations}: {scenario.upper()}, {volatility}%, {horizon}M", end=" ... ")
                    
                    try:
                        if not verbose:
                            f = io.StringIO()
                            with redirect_stdout(f), redirect_stderr(f):
                                result = self.buffer_calculator.calculate_all_methods(scenario, volatility, horizon)
                        else:
                            result = self.buffer_calculator.calculate_all_methods(scenario, volatility, horizon)
                        
                        if 'error' not in result and result.get('methods'):
                            method_data = []
                            for method in result['methods']:
                                if method and 'buffer_price' in method:
                                    method_data.append({
                                        'method_name': method['method'],
                                        'buffer_price': method['buffer_price'],
                                        'buffer_percentage': method['buffer_percentage'],
                                        'final_rate': method['final_rate']
                                    })
                            
                            scenario_data = {
                                'scenario_num': scenario_count,
                                'scenario': scenario,
                                'volatility': volatility,
                                'time_horizon': horizon,
                                'current_rate': result['current_rate'],
                                'current_date': result['current_date'],
                                'methods': method_data,
                                'original_ensemble_avg': result['ensemble_stats']['avg_buffer']
                            }
                            
                            self.all_buffer_results.append(scenario_data)
                            successful_count += 1
                            
                            if verbose:
                                print(f"  ✓ Success - {len(method_data)} methods")
                            else:
                                print(f"SUCCESS ({len(method_data)} methods)")
                        
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            if verbose:
                                print(f"  ✗ Failed: {error_msg}")
                            else:
                                print(f"FAILED: {error_msg}")
                    
                    except Exception as e:
                        if verbose:
                            print(f"  ✗ Error: {e}")
                        else:
                            print(f"ERROR: {e}")
        
        print("\n" + "=" * 70)
        print(f"Buffer calculations complete: {successful_count}/{total_combinations} scenarios successful")
        return successful_count
    
    def setup_volatility_strategies(self):
        """Interactive setup of averaging strategies for each volatility level"""
        
        print("\n" + "=" * 70)
        print("CONFIGURE BUFFER AVERAGING STRATEGIES FOR EACH VOLATILITY LEVEL")
        print("=" * 70)
        
        volatility_levels = [25, 50, 75, 90]
        
        for vol_level in volatility_levels:
            print(f"\nBuffer Volatility Level: {vol_level}%")
            print("-" * 40)
            
            vol_scenarios = [r for r in self.all_buffer_results if r['volatility'] == vol_level]
            print(f"This affects {len(vol_scenarios)} buffer scenarios")
            
            if vol_scenarios:
                sample_methods = len(vol_scenarios[0]['methods'])
                print(f"Each scenario has {sample_methods} calculation methods")
            
            print(f"\nHow should I calculate the final buffer for {vol_level}% volatility scenarios?")
            print("1. Use all methods (average all)")
            print("2. Use top 4 methods (exclude 1 lowest)")
            print("3. Use top 3 methods (exclude 2 lowest)")
            print("4. Use top 2 methods (exclude 3 lowest)")
            print("5. Use only the highest method")
            print("6. Use bottom 3 methods (exclude 2 highest)")
            print("7. Use only specific methods (custom selection)")
            print("8. Use median instead of average")
            print("9. Historical Performance Weighted (Net Coverage)")
            
            while True:
                try:
                    choice = input(f"\nSelect option for {vol_level}% volatility (1-9): ").strip()
                    
                    if choice == '1':
                        self.volatility_strategies[vol_level] = {'type': 'all', 'description': 'All methods average'}
                        break
                    elif choice == '2':
                        self.volatility_strategies[vol_level] = {'type': 'top_n', 'n': 4, 'description': 'Top 4 methods average'}
                        break
                    elif choice == '3':
                        self.volatility_strategies[vol_level] = {'type': 'top_n', 'n': 3, 'description': 'Top 3 methods average'}
                        break
                    elif choice == '4':
                        self.volatility_strategies[vol_level] = {'type': 'top_n', 'n': 2, 'description': 'Top 2 methods average'}
                        break
                    elif choice == '5':
                        self.volatility_strategies[vol_level] = {'type': 'top_n', 'n': 1, 'description': 'Highest method only'}
                        break
                    elif choice == '6':
                        self.volatility_strategies[vol_level] = {'type': 'bottom_n', 'n': 3, 'description': 'Bottom 3 methods average'}
                        break
                    elif choice == '7':
                        if vol_scenarios:
                            print(f"\nAvailable methods in these scenarios:")
                            sample_methods = vol_scenarios[0]['methods']
                            for i, method in enumerate(sample_methods, 1):
                                print(f"  {i}. {method['method_name']}")
                            
                            method_indices = input("\nEnter method numbers to include (e.g., 1,3,5): ").strip()
                            try:
                                indices = [int(x.strip()) - 1 for x in method_indices.split(',')]
                                valid_indices = [i for i in indices if 0 <= i < len(sample_methods)]
                                if valid_indices:
                                    self.volatility_strategies[vol_level] = {
                                        'type': 'custom', 
                                        'indices': valid_indices,
                                        'description': f'Custom methods: {[sample_methods[i]["method_name"].split()[0] for i in valid_indices]}'
                                    }
                                    break
                                else:
                                    print("Invalid method numbers. Please try again.")
                            except ValueError:
                                print("Invalid format. Please enter numbers separated by commas.")
                        else:
                            print("No scenarios available for custom selection.")
                    elif choice == '8':
                        self.volatility_strategies[vol_level] = {'type': 'median', 'description': 'Median of all methods'}
                        break
                    elif choice == '9':
                        self.volatility_strategies[vol_level] = {'type': 'historical_performance_weighted', 'description': 'Historical Performance Weighted (Net Coverage)'}
                        break
                    else:
                        print("Please enter a number between 1 and 9")
                        
                except KeyboardInterrupt:
                    print("\nSetup cancelled.")
                    return False
            
            print(f"✓ {vol_level}% volatility: {self.volatility_strategies[vol_level]['description']}")
        
        return True
    
    def apply_custom_averaging(self):
        """Apply the user-defined averaging strategies to calculate final buffers"""
        
        print("\n" + "=" * 70)
        print("APPLYING CUSTOM BUFFER AVERAGING STRATEGIES")
        print("=" * 70)
        
        final_buffer_results = []
        
        for result in self.all_buffer_results:
            vol_level = result['volatility']
            strategy = self.volatility_strategies[vol_level]
            methods = result['methods']
            
            # Sort methods by buffer price (descending)
            sorted_methods = sorted(methods, key=lambda x: x['buffer_price'], reverse=True)
            
            if strategy['type'] == 'all':
                selected_methods = methods
                
            elif strategy['type'] == 'top_n':
                n = min(strategy['n'], len(sorted_methods))
                selected_methods = sorted_methods[:n]
                
            elif strategy['type'] == 'bottom_n':
                n = min(strategy['n'], len(sorted_methods))
                selected_methods = sorted_methods[-n:]
                
            elif strategy['type'] == 'custom':
                indices = strategy['indices']
                selected_methods = [methods[i] for i in indices if i < len(methods)]
                
            elif strategy['type'] == 'median':
                buffer_prices = [m['buffer_price'] for m in methods]
                median_buffer = np.median(buffer_prices)
                selected_methods = [{'buffer_price': median_buffer, 'method_name': 'Median'}]
                
            elif strategy['type'] == 'historical_performance_weighted':
                # Test each method's historical performance using net coverage
                print(f"  Testing historical performance for {result['scenario']} {result['volatility']}% {result['time_horizon']}M...")
                
                method_performance = []
                time_horizon = result['time_horizon']
                
                for i, method in enumerate(methods, 1):
                    buffer_pct = method['buffer_percentage']
                    method_name = method['method_name']
                    
                    print(f"    [{i}/{len(methods)}] Testing {method_name}: {buffer_pct:.2f}%", end=" ... ")
                    
                    try:
                        # Suppress output from historical analysis
                        f = io.StringIO()
                        with redirect_stdout(f), redirect_stderr(f):
                            historical_analysis = self.run_historical_analysis_for_buffer(buffer_pct, time_horizon)
                        
                        if historical_analysis and 'net_containment_rate' in historical_analysis:
                            net_coverage = historical_analysis['net_containment_rate']
                            method_performance.append({
                                'method': method,
                                'net_coverage': net_coverage,
                                'buffer_percentage': buffer_pct
                            })
                            print(f"Coverage: {net_coverage:.1f}%")
                        else:
                            print("FAILED")
                            
                    except Exception as e:
                        print(f"ERROR: {e}")
                
                if method_performance:
                    # Calculate net coverage weighted buffer
                    weighted_sum = 0
                    total_weight = 0
                    
                    print(f"    Calculating performance-weighted buffer:")
                    for perf in method_performance:
                        buffer_pct = perf['buffer_percentage']
                        net_coverage = perf['net_coverage'] / 100  # Convert to decimal
                        weighted_buffer = buffer_pct * net_coverage
                        
                        weighted_sum += weighted_buffer
                        total_weight += net_coverage
                        
                        print(f"      {perf['method']['method_name']}: {buffer_pct:.2f}% × {net_coverage:.3f} = {weighted_buffer:.4f}")
                    
                    if total_weight > 0:
                        performance_weighted_buffer_pct = weighted_sum / total_weight
                        # Convert percentage back to buffer price
                        current_rate = result['current_rate']
                        performance_weighted_buffer_price = (performance_weighted_buffer_pct / 100) * current_rate
                        
                        selected_methods = [{
                            'buffer_price': performance_weighted_buffer_price, 
                            'method_name': f'Performance Weighted ({len(method_performance)} methods)'
                        }]
                        
                        print(f"    Final performance-weighted buffer: {performance_weighted_buffer_pct:.2f}%")
                    else:
                        print(f"    ERROR: No valid performance data, falling back to simple average")
                        selected_methods = methods
                else:
                    print(f"    ERROR: No methods tested successfully, falling back to simple average")
                    selected_methods = methods
            
            # Calculate final buffer based on selected methods
            if selected_methods:
                if strategy['type'] == 'median':
                    final_buffer = selected_methods[0]['buffer_price']
                elif strategy['type'] == 'historical_performance_weighted':
                    final_buffer = selected_methods[0]['buffer_price']
                else:
                    final_buffer = np.mean([m['buffer_price'] for m in selected_methods])
                
                final_percentage = (final_buffer / result['current_rate']) * 100
                
                final_buffer_results.append({
                    'scenario_num': result['scenario_num'],
                    'scenario': result['scenario'],
                    'volatility': result['volatility'],
                    'time_horizon': result['time_horizon'],
                    'final_buffer_price': final_buffer,
                    'final_buffer_percentage': final_percentage,
                    'methods_used': len(selected_methods),
                    'strategy': strategy['description'],
                    'original_ensemble': result['original_ensemble_avg'],
                    'current_rate': result['current_rate']
                })
        
        return final_buffer_results
    
    def run_historical_analysis_for_buffer(self, buffer_percentage, historical_period_months):
        """Run historical analysis for a single buffer percentage with specified period"""
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - relativedelta(years=historical_period_months)
            
            df = self.volatility_tester.fetch_exchange_rate_data(
                'FXUSDCAD',
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return None
            
            # Calculate movements
            movements_df = self.volatility_tester.calculate_rolling_movements(df, historical_period_months)
            
            if movements_df.empty:
                return None
            
            # Analyze net movement containment
            net_results, movements_with_analysis = self.volatility_tester.analyze_containment(
                movements_df, buffer_percentage, historical_period_months
            )
            
            # Analyze max-min volatility containment
            volatility_stats, movements_with_volatility = self.volatility_tester.analyze_max_min_volatility_containment(
                movements_with_analysis, buffer_percentage
            )
            
            # Calculate detailed failure counts
            appreciation_movements = movements_with_analysis[movements_with_analysis['percent_movement'] < 0]
            depreciation_movements = movements_with_analysis[movements_with_analysis['percent_movement'] > 0]
            
            appreciation_failures = len(appreciation_movements[appreciation_movements['percent_movement'].abs() > buffer_percentage])
            depreciation_failures = len(depreciation_movements[depreciation_movements['percent_movement'] > buffer_percentage])
            volatility_failures = len(movements_with_volatility[movements_with_volatility['max_min_volatility'] > buffer_percentage])
            
            return {
                'net_containment_rate': net_results['containment_rate'],
                'net_failure_rate': net_results['failure_rate'],
                'max_min_containment_rate': volatility_stats['volatility_containment_rate'],
                'max_min_failure_rate': volatility_stats['volatility_failure_rate'],
                'total_movements': net_results['total_movements'],
                'appreciation_failures': appreciation_failures,
                'depreciation_failures': depreciation_failures,
                'volatility_failures': volatility_failures,
                'appreciation_total': len(appreciation_movements),
                'depreciation_total': len(depreciation_movements)
            }
            
        except Exception as e:
            print(f"Error in historical analysis: {e}")
            return None
    
    def run_matched_period_analysis(self):
        """Run historical analysis with matched time periods"""
        
        print("\n" + "=" * 70)
        print("RUNNING MATCHED-PERIOD HISTORICAL ANALYSIS")
        print("=" * 70)
        print("Matching buffer time horizons with historical analysis periods:")
        print("• 1-month buffers → 1-month historical analysis")
        print("• 3-month buffers → 3-month historical analysis")
        print("• 6-month buffers → 6-month historical analysis")
        print("• 12-month buffers → 12-month historical analysis")
        
        final_buffer_results = self.apply_custom_averaging()
        
        if not final_buffer_results:
            print("No buffer results to analyze")
            return None
        
        # Organize results by time horizon and scenario
        results_by_period = {}
        
        for time_horizon in [1, 3, 6, 12]:
            print(f"\nProcessing {time_horizon}-month scenarios...")
            
            # Get buffers for this time horizon
            period_buffers = [r for r in final_buffer_results if r['time_horizon'] == time_horizon]
            
            if not period_buffers:
                continue
                
            results_by_period[time_horizon] = {}
            
            for scenario in ['depreciating', 'appreciating']:
                scenario_buffers = [r for r in period_buffers if r['scenario'] == scenario]
                results_by_period[time_horizon][scenario] = {}
                
                for buffer_result in scenario_buffers:
                    volatility_level = buffer_result['volatility']
                    buffer_percentage = buffer_result['final_buffer_percentage']
                    
                    print(f"  Analyzing {scenario} {volatility_level}% buffer: {buffer_percentage:.2f}%", end=" ... ")
                    
                    # Run historical analysis with matching time period
                    f = io.StringIO()
                    with redirect_stdout(f), redirect_stderr(f):
                        historical_analysis = self.run_historical_analysis_for_buffer(buffer_percentage, time_horizon)
                    
                    if historical_analysis:
                        results_by_period[time_horizon][scenario][volatility_level] = {
                            'buffer_percentage': buffer_percentage,
                            'historical_analysis': historical_analysis
                        }
                        print("SUCCESS")
                    else:
                        print("FAILED")
        
        self.final_client_results = results_by_period
        # Store original results for rollback capability
        self.original_client_results = copy.deepcopy(results_by_period)
        return results_by_period
    
    def create_client_presentation_charts(self):
        """Create client-ready presentation charts"""
        
        print("\n" + "=" * 80)
        print("CREATING CLIENT PRESENTATION CHARTS")
        print("=" * 80)
        
        if not self.final_client_results:
            print("No results available for chart creation")
            return
        
        # Create depreciation chart
        print("\nDEPRECIATION BUFFERS & HISTORICAL COVERAGE")
        print("=" * 95)
        print(f"{'Risk':<5} {'1 Month':<18} {'3 Months':<18} {'6 Months':<18} {'12 Months':<18}")
        print(f"{'Level':<5} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18}")
        print("-" * 95)
        
        for vol_level in [25, 50, 75, 90]:
            row_data = [f"{vol_level}%"]
            
            for time_horizon in [1, 3, 6, 12]:
                if (time_horizon in self.final_client_results and 
                    'depreciating' in self.final_client_results[time_horizon] and
                    vol_level in self.final_client_results[time_horizon]['depreciating']):
                    
                    data = self.final_client_results[time_horizon]['depreciating'][vol_level]
                    buf_pct = data['buffer_percentage']
                    net_pct = data['historical_analysis']['net_containment_rate']
                    max_pct = data['historical_analysis']['max_min_containment_rate']
                    
                    cell_text = f"{buf_pct:.2f} {net_pct:.0f} {max_pct:.0f}"
                else:
                    cell_text = "N/A   N/A  N/A"
                
                row_data.append(cell_text)
            
            print(f"{row_data[0]:<5} {row_data[1]:<18} {row_data[2]:<18} {row_data[3]:<18} {row_data[4]:<18}")
        
        # Create appreciation chart
        print(f"\n\nAPPRECIATION BUFFERS & HISTORICAL COVERAGE")
        print("=" * 95)
        print(f"{'Risk':<5} {'1 Month':<18} {'3 Months':<18} {'6 Months':<18} {'12 Months':<18}")
        print(f"{'Level':<5} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18} {'Buf% Net% Max%':<18}")
        print("-" * 95)
        
        for vol_level in [25, 50, 75, 90]:
            row_data = [f"{vol_level}%"]
            
            for time_horizon in [1, 3, 6, 12]:
                if (time_horizon in self.final_client_results and 
                    'appreciating' in self.final_client_results[time_horizon] and
                    vol_level in self.final_client_results[time_horizon]['appreciating']):
                    
                    data = self.final_client_results[time_horizon]['appreciating'][vol_level]
                    buf_pct = data['buffer_percentage']
                    net_pct = data['historical_analysis']['net_containment_rate']
                    max_pct = data['historical_analysis']['max_min_containment_rate']
                    
                    cell_text = f"{buf_pct:.2f} {net_pct:.0f} {max_pct:.0f}"
                else:
                    cell_text = "N/A   N/A  N/A"
                
                row_data.append(cell_text)
            
            print(f"{row_data[0]:<5} {row_data[1]:<18} {row_data[2]:<18} {row_data[3]:<18} {row_data[4]:<18}")
        
        print(f"\nLegend:")
        print(f"Buf% = Buffer Percentage")
        print(f"Net% = Net Movement Coverage Rate")
        print(f"Max% = Max-Min Volatility Coverage Rate")
    
    def optimize_buffer_for_coverage_target(self, original_buffer_pct, time_horizon, coverage_type, target_coverage, increment=0.1, max_multiplier=3.0):
        """Optimize buffer by incrementally increasing until target coverage is achieved"""
        
        coverage_name = "Net" if coverage_type == "net" else "Max-Min"
        print(f"\n    🔧 OPTIMIZING BUFFER for {target_coverage}% {coverage_name} coverage...")
        print(f"    Original buffer: {original_buffer_pct:.2f}%")
        
        current_buffer = original_buffer_pct
        max_buffer = original_buffer_pct * max_multiplier
        iteration = 0
        max_iterations = int((max_buffer - original_buffer_pct) / increment) + 1
        
        while current_buffer <= max_buffer and iteration < max_iterations:
            iteration += 1
            
            # Test current buffer
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                test_result = self.run_historical_analysis_for_buffer(current_buffer, time_horizon)
            
            if test_result:
                max_min_coverage = test_result['max_min_containment_rate']
                net_coverage = test_result['net_containment_rate']
                
                current_coverage = net_coverage if coverage_type == "net" else max_min_coverage
                
                print(f"    Test {iteration}: {current_buffer:.2f}% → Net: {net_coverage:.1f}%, Max-Min: {max_min_coverage:.1f}%", end="")
                
                if current_coverage >= target_coverage:
                    print(" ✓ TARGET ACHIEVED!")
                    return {
                        'optimized_buffer': current_buffer,
                        'original_buffer': original_buffer_pct,
                        'iterations': iteration,
                        'final_net_coverage': net_coverage,
                        'final_max_min_coverage': max_min_coverage,
                        'target_coverage': target_coverage,
                        'coverage_type': coverage_type,
                        'historical_analysis': test_result
                    }
                else:
                    print("")
                    current_buffer += increment
            else:
                print(f"    Test {iteration}: {current_buffer:.2f}% → FAILED")
                current_buffer += increment
        
        print(f"    ❌ Could not achieve {target_coverage}% {coverage_name} coverage within {max_multiplier}x original buffer limit")
        return None
    
    def enhanced_optimization_analysis(self):
        """Enhanced optimization with individual scenario selection and rollback options"""
        
        print(f"\n" + "=" * 70)
        print("ENHANCED BUFFER OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        # Find scenarios that could benefit from optimization
        optimization_candidates = []
        
        for time_horizon in [1, 3, 6, 12]:
            if time_horizon not in self.final_client_results:
                continue
                
            for scenario in ['depreciating', 'appreciating']:
                if scenario not in self.final_client_results[time_horizon]:
                    continue
                    
                for vol_level in [25, 50, 75, 90]:
                    if vol_level not in self.final_client_results[time_horizon][scenario]:
                        continue
                        
                    data = self.final_client_results[time_horizon][scenario][vol_level]
                    hist = data['historical_analysis']
                    
                    # Show all scenarios, not just poor performers
                    optimization_candidates.append({
                        'time_horizon': time_horizon,
                        'scenario': scenario,
                        'vol_level': vol_level,
                        'buffer_percentage': data['buffer_percentage'],
                        'net_coverage': hist['net_containment_rate'],
                        'max_min_coverage': hist['max_min_containment_rate']
                    })
        
        if not optimization_candidates:
            print("No scenarios available for optimization.")
            return {}
        
        print(f"Available scenarios for optimization:")
        print(f"{'#':<3} {'Scenario':<25} {'Buffer%':<10} {'Net%':<8} {'Max-Min%':<10}")
        print("-" * 65)
        
        for i, candidate in enumerate(optimization_candidates, 1):
            scenario_label = f"{candidate['time_horizon']}M {candidate['scenario'][:4]} {candidate['vol_level']}%"
            print(f"{i:<3} {scenario_label:<25} {candidate['buffer_percentage']:.2f}%{'':<4} {candidate['net_coverage']:.1f}%{'':<3} {candidate['max_min_coverage']:.1f}%")
        
        # Ask optimization approach
        print(f"\nOptimization options:")
        print("A. Optimize all scenarios with Max-Min coverage < 1%")
        print("B. Select individual scenarios to optimize")
        print("C. Skip optimization")
        print("R. Rollback previous optimizations")
        
        while True:
            choice = input(f"\nSelect option (A/B/C/R): ").strip().upper()
            if choice in ['A', 'B', 'C', 'R']:
                break
            print("Please enter A, B, C, or R")
        
        if choice == 'C':
            print("Optimization skipped.")
            return {}
        
        if choice == 'R':
            return self.handle_rollback()
        
        if choice == 'A':
            # Optimize all poor performers
            poor_performers = [c for c in optimization_candidates if c['max_min_coverage'] < 1.0]
            if not poor_performers:
                print("No scenarios with Max-Min coverage < 1% found.")
                return {}
            
            selected_scenarios = poor_performers
            coverage_type = "max_min"
            target_coverage = 1.0
            return self.run_bulk_optimization(selected_scenarios, coverage_type, target_coverage)
            
        elif choice == 'B':
            # Individual selection
            selected_scenarios = self.select_individual_scenarios(optimization_candidates)
            if not selected_scenarios:
                return {}
            
            return self.optimize_selected_scenarios(selected_scenarios)
    
    def select_individual_scenarios(self, candidates):
        """Let user select individual scenarios to optimize"""
        
        print(f"\nSelect scenarios to optimize:")
        print("Enter scenario numbers separated by commas (e.g., 1,5,12)")
        print("Or enter 'all' to select all scenarios")
        
        while True:
            selection = input("Scenarios to optimize: ").strip()
            
            if selection.lower() == 'all':
                return candidates
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]
                
                if selected:
                    print(f"\nSelected {len(selected)} scenarios:")
                    for scenario in selected:
                        scenario_label = f"{scenario['time_horizon']}M {scenario['scenario'][:4]} {scenario['vol_level']}%"
                        print(f"  {scenario_label}: {scenario['buffer_percentage']:.2f}% buffer")
                    return selected
                else:
                    print("No valid scenarios selected. Please try again.")
                    
            except ValueError:
                print("Invalid format. Please enter numbers separated by commas.")
    
    def optimize_selected_scenarios(self, selected_scenarios):
        """Optimize individually selected scenarios with custom targets"""
        
        optimization_results = {}
        
        # Get global settings first
        print(f"\n" + "=" * 50)
        print("OPTIMIZATION SETTINGS")
        print("=" * 50)
        
        increment, max_multiplier = self.ask_for_optimization_increment_settings()
        
        for i, scenario in enumerate(selected_scenarios, 1):
            scenario_label = f"{scenario['time_horizon']}M {scenario['scenario']} {scenario['vol_level']}%"
            
            print(f"\n[{i}/{len(selected_scenarios)}] Optimizing {scenario_label}")
            print(f"Current: {scenario['buffer_percentage']:.2f}% buffer, {scenario['net_coverage']:.1f}% net, {scenario['max_min_coverage']:.1f}% max-min")
            
            # Ask what to optimize for
            print(f"\nWhat would you like to optimize for?")
            print(f"1. Net coverage (currently {scenario['net_coverage']:.1f}%)")
            print(f"2. Max-Min coverage (currently {scenario['max_min_coverage']:.1f}%)")
            print(f"3. Skip this scenario")
            
            while True:
                opt_choice = input("Select option (1/2/3): ").strip()
                if opt_choice in ['1', '2', '3']:
                    break
                print("Please enter 1, 2, or 3")
            
            if opt_choice == '3':
                print(f"Skipping {scenario_label}")
                continue
            
            coverage_type = "net" if opt_choice == '1' else "max_min"
            coverage_name = "Net" if opt_choice == '1' else "Max-Min"
            current_coverage = scenario['net_coverage'] if opt_choice == '1' else scenario['max_min_coverage']
            
            # Ask for target coverage
            while True:
                try:
                    target_input = input(f"Enter target {coverage_name} coverage % (current: {current_coverage:.1f}%): ").strip()
                    target_coverage = float(target_input)
                    
                    if target_coverage > current_coverage:
                        break
                    else:
                        print(f"Target must be higher than current {current_coverage:.1f}%")
                except ValueError:
                    print("Please enter a valid number")
            
            # Run optimization
            print(f"\nOptimizing {scenario_label} for {target_coverage}% {coverage_name} coverage...")
            
            optimization_result = self.optimize_buffer_for_coverage_target(
                scenario['buffer_percentage'],
                scenario['time_horizon'],
                coverage_type,
                target_coverage,
                increment,
                max_multiplier
            )
            
            if optimization_result:
                # Update results
                key = f"{scenario['time_horizon']}_{scenario['scenario']}_{scenario['vol_level']}"
                optimization_results[key] = optimization_result
                
                # Update final_client_results with optimized values
                self.final_client_results[scenario['time_horizon']][scenario['scenario']][scenario['vol_level']] = {
                    'buffer_percentage': optimization_result['optimized_buffer'],
                    'historical_analysis': optimization_result['historical_analysis']
                }
                
                print(f"✓ {scenario_label} optimized successfully!")
            else:
                print(f"✗ {scenario_label} optimization failed")
        
        return optimization_results
    
    def run_bulk_optimization(self, scenarios, coverage_type, target_coverage):
        """Run bulk optimization for multiple scenarios"""
        
        print(f"\nOptimizing {len(scenarios)} scenarios for {target_coverage}% {'Net' if coverage_type == 'net' else 'Max-Min'} coverage...")
        
        increment, max_multiplier = self.ask_for_optimization_increment_settings()
        optimization_results = {}
        
        for i, scenario in enumerate(scenarios, 1):
            scenario_label = f"{scenario['time_horizon']}M {scenario['scenario']} {scenario['vol_level']}%"
            
            print(f"\n[{i}/{len(scenarios)}] Optimizing {scenario_label}")
            
            optimization_result = self.optimize_buffer_for_coverage_target(
                scenario['buffer_percentage'],
                scenario['time_horizon'],
                coverage_type,
                target_coverage,
                increment,
                max_multiplier
            )
            
            if optimization_result:
                key = f"{scenario['time_horizon']}_{scenario['scenario']}_{scenario['vol_level']}"
                optimization_results[key] = optimization_result
                
                # Update final_client_results
                self.final_client_results[scenario['time_horizon']][scenario['scenario']][scenario['vol_level']] = {
                    'buffer_percentage': optimization_result['optimized_buffer'],
                    'historical_analysis': optimization_result['historical_analysis']
                }
        
        return optimization_results
    
    def handle_rollback(self):
        """Handle rollback of optimizations"""
        
        if not hasattr(self, 'original_client_results') or not self.original_client_results:
            print("No original results available for rollback.")
            return {}
        
        print(f"\nROLLBACK OPTIONS:")
        print("1. Rollback all optimizations")
        print("2. Select specific scenarios to rollback")
        print("3. Cancel rollback")
        
        while True:
            choice = input("Select rollback option (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Please enter 1, 2, or 3")
        
        if choice == '3':
            print("Rollback cancelled.")
            return {}
        
        if choice == '1':
            # Rollback everything
            self.final_client_results = copy.deepcopy(self.original_client_results)
            print("✓ All optimizations rolled back to original values")
            return {'rollback': 'all'}
        
        elif choice == '2':
            # Individual rollback
            return self.select_scenarios_for_rollback()
    
    def select_scenarios_for_rollback(self):
        """Select specific scenarios to rollback"""
        
        # Find scenarios that have been modified
        rollback_candidates = []
        
        for time_horizon in [1, 3, 6, 12]:
            if time_horizon not in self.final_client_results or time_horizon not in self.original_client_results:
                continue
                
            for scenario in ['depreciating', 'appreciating']:
                if scenario not in self.final_client_results[time_horizon] or scenario not in self.original_client_results[time_horizon]:
                    continue
                    
                for vol_level in [25, 50, 75, 90]:
                    if vol_level not in self.final_client_results[time_horizon][scenario] or vol_level not in self.original_client_results[time_horizon][scenario]:
                        continue
                    
                    current_buffer = self.final_client_results[time_horizon][scenario][vol_level]['buffer_percentage']
                    original_buffer = self.original_client_results[time_horizon][scenario][vol_level]['buffer_percentage']
                    
                    if abs(current_buffer - original_buffer) > 0.01:  # If changed
                        rollback_candidates.append({
                            'time_horizon': time_horizon,
                            'scenario': scenario,
                            'vol_level': vol_level,
                            'current_buffer': current_buffer,
                            'original_buffer': original_buffer
                        })
        
        if not rollback_candidates:
            print("No optimized scenarios found to rollback.")
            return {}
        
        print(f"\nScenarios available for rollback:")
        print(f"{'#':<3} {'Scenario':<25} {'Current%':<10} {'Original%':<10} {'Change':<10}")
        print("-" * 70)
        
        for i, candidate in enumerate(rollback_candidates, 1):
            scenario_label = f"{candidate['time_horizon']}M {candidate['scenario'][:4]} {candidate['vol_level']}%"
            change = candidate['current_buffer'] - candidate['original_buffer']
            print(f"{i:<3} {scenario_label:<25} {candidate['current_buffer']:.2f}%{'':<4} {candidate['original_buffer']:.2f}%{'':<4} {change:+.2f}%")
        
        while True:
            selection = input(f"\nSelect scenarios to rollback (e.g., 1,3,5) or 'all': ").strip()
            
            if selection.lower() == 'all':
                selected = rollback_candidates
                break
            
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected = [rollback_candidates[i] for i in indices if 0 <= i < len(rollback_candidates)]
                
                if selected:
                    break
                else:
                    print("No valid scenarios selected. Please try again.")
                    
            except ValueError:
                print("Invalid format. Please enter numbers separated by commas.")
        
        # Perform rollback
        rollback_count = 0
        for candidate in selected:
            time_horizon = candidate['time_horizon']
            scenario = candidate['scenario']
            vol_level = candidate['vol_level']
            
            # Restore original values
            self.final_client_results[time_horizon][scenario][vol_level] = \
                self.original_client_results[time_horizon][scenario][vol_level].copy()
            
            scenario_label = f"{time_horizon}M {scenario[:4]} {vol_level}%"
            print(f"✓ Rolled back {scenario_label}: {candidate['current_buffer']:.2f}% → {candidate['original_buffer']:.2f}%")
            rollback_count += 1
        
        print(f"\n✓ Rolled back {rollback_count} scenarios to original values")
        return {'rollback': f'{rollback_count} scenarios'}
    
    def ask_for_optimization_increment_settings(self):
        """Ask user for optimization increment settings"""
        
        # Ask for increment
        while True:
            try:
                increment_input = input("Enter buffer increment size (e.g., 0.1 for 0.1%): ").strip()
                increment = float(increment_input)
                if increment > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Ask for maximum multiplier
        while True:
            try:
                max_input = input("Enter maximum buffer multiplier (e.g., 3.0 for 3x original): ").strip()
                max_multiplier = float(max_input)
                if max_multiplier > 1:
                    break
                else:
                    print("Please enter a number greater than 1.")
            except ValueError:
                print("Please enter a valid number.")
        
        return increment, max_multiplier
    
    def display_optimization_summary(self, optimization_results):
        """Display optimization summary with enhanced formatting"""
        
        if not optimization_results:
            return
            
        if 'rollback' in optimization_results:
            print(f"\n✓ Rollback completed: {optimization_results['rollback']}")
            return
        
        print(f"\n" + "=" * 90)
        print("OPTIMIZATION SUMMARY")
        print("=" * 90)
        
        print(f"{'Scenario':<25} {'Original':<10} {'Optimized':<10} {'Target':<12} {'Coverage':<8} {'Iterations':<10}")
        print("-" * 90)
        
        successful_optimizations = 0
        
        for key, result in optimization_results.items():
            if key == 'rollback':
                continue
                
            parts = key.split('_')
            time_horizon = parts[0]
            scenario = parts[1]
            vol_level = parts[2]
            
            scenario_label = f"{time_horizon}M {scenario[:4].upper()} {vol_level}%"
            original_buf = f"{result['original_buffer']:.2f}%"
            optimized_buf = f"{result['optimized_buffer']:.2f}%"
            target_str = f"{result['target_coverage']:.1f}%"
            iterations = str(result['iterations'])
            
            # Show final achieved coverage
            if result['coverage_type'] == 'net':
                final_coverage = f"{result['final_net_coverage']:.1f}%"
            else:
                final_coverage = f"{result['final_max_min_coverage']:.1f}%"
            
            print(f"{scenario_label:<25} {original_buf:<10} {optimized_buf:<10} {target_str:<12} {final_coverage:<8} {iterations:<10}")
            successful_optimizations += 1
        
        print(f"\n✓ Successfully optimized {successful_optimizations} scenarios")
    
    def create_detailed_analysis_summary(self):
        """Create detailed analysis summary for internal use"""
        
        print("\n" + "=" * 100)
        print("DETAILED ANALYSIS SUMMARY")
        print("=" * 100)
        
        if not self.final_client_results:
            print("No results available")
            return
        
        optimization_needed = []
        
        for time_horizon in [1, 3, 6, 12]:
            if time_horizon not in self.final_client_results:
                continue
                
            print(f"\n{time_horizon}-MONTH PERIOD ANALYSIS")
            print("-" * 60)
            
            for scenario in ['depreciating', 'appreciating']:
                if scenario not in self.final_client_results[time_horizon]:
                    continue
                    
                print(f"\n{scenario.upper()} Scenario:")
                scenario_data = self.final_client_results[time_horizon][scenario]
                
                for vol_level in [25, 50, 75, 90]:
                    if vol_level not in scenario_data:
                        continue
                        
                    data = scenario_data[vol_level]
                    buf_pct = data['buffer_percentage']
                    hist = data['historical_analysis']
                    
                    print(f"  {vol_level}% Risk Level:")
                    print(f"    Buffer: {buf_pct:.2f}%")
                    print(f"    Net Coverage: {hist['net_containment_rate']:.1f}% ({hist['total_movements']} movements)")
                    print(f"    Max-Min Coverage: {hist['max_min_containment_rate']:.1f}%")
                    
                    if scenario == 'depreciating':
                        print(f"    Depreciation Failures: {hist['depreciation_failures']}")
                        print(f"    Appreciation Failures: {hist['appreciation_failures']}")
                    else:
                        print(f"    Appreciation Failures: {hist['appreciation_failures']}")
                        print(f"    Depreciation Failures: {hist['depreciation_failures']}")
                    
                    print(f"    Volatility Range Failures: {hist['volatility_failures']}")
                    
                    # Track scenarios with poor coverage
                    if hist['max_min_containment_rate'] < 1.0:
                        optimization_needed.append(f"{time_horizon}M {scenario} {vol_level}%")
        
        # Show optimization opportunities
        if optimization_needed:
            print(f"\n" + "=" * 60)
            print("OPTIMIZATION OPPORTUNITIES")
            print("=" * 60)
            print(f"Scenarios with Max-Min coverage < 1.0% (may benefit from optimization):")
            for scenario in optimization_needed:
                print(f"  • {scenario}")
        
        return len(optimization_needed) > 0
    
    def save_client_results_to_csv(self):
        """Save client-ready results to CSV"""
        try:
            # Create comprehensive CSV data
            csv_data = []
            
            for time_horizon in [1, 3, 6, 12]:
                if time_horizon not in self.final_client_results:
                    continue
                    
                for scenario in ['depreciating', 'appreciating']:
                    if scenario not in self.final_client_results[time_horizon]:
                        continue
                        
                    for vol_level in [25, 50, 75, 90]:
                        if vol_level not in self.final_client_results[time_horizon][scenario]:
                            continue
                            
                        data = self.final_client_results[time_horizon][scenario][vol_level]
                        hist = data['historical_analysis']
                        
                        csv_row = {
                            'Time_Horizon_Months': time_horizon,
                            'Scenario': scenario,
                            'Volatility_Level': vol_level,
                            'Buffer_Percentage': data['buffer_percentage'],
                            'Net_Coverage_Rate': hist['net_containment_rate'],
                            'Max_Min_Coverage_Rate': hist['max_min_containment_rate'],
                            'Net_Failure_Rate': hist['net_failure_rate'],
                            'Max_Min_Failure_Rate': hist['max_min_failure_rate'],
                            'Total_Movements_Analyzed': hist['total_movements'],
                            'Appreciation_Failures': hist['appreciation_failures'],
                            'Depreciation_Failures': hist['depreciation_failures'],
                            'Volatility_Failures': hist['volatility_failures'],
                            'Appreciation_Total': hist['appreciation_total'],
                            'Depreciation_Total': hist['depreciation_total']
                        }
                        
                        csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"client_ready_buffer_analysis_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\nClient results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"\nError saving CSV: {e}")
            return None
    
    def ask_for_save_options(self):
        """Ask user about saving options"""
        while True:
            save_input = input("\nSave client-ready results to CSV file? (y/n): ").strip().lower()
            if save_input in ['y', 'yes', 'n', 'no']:
                return save_input in ['y', 'yes']
            print("Please enter 'y' or 'n'")
    
    def ask_for_another_calculation(self):
        """Ask if user wants to run another calculation"""
        while True:
            another = input("\nWould you like to run another calculation with different settings? (y/n): ").strip().lower()
            if another in ['y', 'yes', 'n', 'no']:
                return another in ['y', 'yes']
            print("Please enter 'y' or 'n'")

def main():
    """Main function to run the CAD Buffer Optimizer Pro"""
    print("CAD BUFFER OPTIMIZER PRO - COMPLETE ANALYSIS SYSTEM")
    print("=" * 70)
    print("This system will:")
    print("1. Calculate buffers for all 32 scenarios with your custom averaging")
    print("2. Run matched-period historical analysis:")
    print("   • 1-month buffers tested with 1-month historical periods")
    print("   • 3-month buffers tested with 3-month historical periods")
    print("   • 6-month buffers tested with 6-month historical periods")  
    print("   • 12-month buffers tested with 12-month historical periods")
    print("3. Generate client presentation charts showing buffer & coverage rates")
    print("4. Optimize buffers with poor coverage using iterative enhancement")
    
    analyzer = ClientReadyBufferAnalyzer()
    
    while True:
        try:
            # Step 1: Run buffer calculations
            print(f"\nSTEP 1: Running buffer calculations...")
            verbose_choice = input("Show detailed output for buffer calculations? (y/n): ").strip().lower()
            verbose = verbose_choice in ['y', 'yes']
            
            successful_runs = analyzer.run_all_buffer_scenarios(verbose=verbose)
            
            if successful_runs == 0:
                print("No successful buffer calculations. Please check your buffer script.")
                return
            
            # Step 2: Configure averaging strategies
            print(f"\nSTEP 2: Configure buffer averaging strategies...")
            if not analyzer.setup_volatility_strategies():
                print("Configuration cancelled.")
                return
            
            # Step 3: Run matched-period analysis
            print(f"\nSTEP 3: Running matched-period historical analysis...")
            results = analyzer.run_matched_period_analysis()
            
            if not results:
                print("Historical analysis failed.")
                return
            
            # Step 4: Create client presentation charts
            print(f"\nSTEP 4: Creating client presentation charts...")
            analyzer.create_client_presentation_charts()
            
            # Step 5: Create detailed summary and check for optimization opportunities
            print(f"\nSTEP 5: Creating detailed analysis summary...")
            needs_optimization = analyzer.create_detailed_analysis_summary()
            
            # Step 6: Ask about optimization if needed
            optimization_results = {}
            if needs_optimization:
                optimize_choice = input(f"\nWould you like to run buffer optimization for scenarios with poor Max-Min coverage? (y/n): ").strip().lower()
                if optimize_choice in ['y', 'yes']:
                    optimization_results = analyzer.enhanced_optimization_analysis()
                    
                    if optimization_results:
                        print(f"\nSTEP 6A: Optimization completed, updating charts...")
                        analyzer.display_optimization_summary(optimization_results)
                        analyzer.create_client_presentation_charts()  # Update charts with optimized values
            
            # Step 7: Save options
            if analyzer.ask_for_save_options():
                analyzer.save_client_results_to_csv()
            
            print(f"\nClient-ready analysis completed successfully!")
            print(f"   {successful_runs} buffer scenarios processed")
            print(f"   Matched-period historical analysis completed")
            print(f"   Client presentation charts generated")
            if optimization_results:
                print(f"   {len(optimization_results)} scenarios optimized for better coverage")
            
            # Ask if user wants another calculation
            if not analyzer.ask_for_another_calculation():
                print("\nThank you for using the Client-Ready Buffer Analysis System!")
                break
                
        except KeyboardInterrupt:
            print(f"\nAnalysis cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please check your scripts and try again.")
            
            retry = input("\nWould you like to try again? (y/n): ").strip().lower()
            if retry not in ['y', 'yes']:
                break

if __name__ == "__main__":
    main()