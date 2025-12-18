import os
import sys
import json
import random
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path

# --- RICH LIBRARY SETUP ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def __init__(self): pass
        def print(self, *args, **kwargs):
            print(*(str(arg).replace('[bold red]', '').replace('[/bold red]', '').replace('[bold green]', '').replace('[/bold green]', '') for arg in args))
        def input(self, prompt=""): return input(prompt)
    class Panel:
        def __init__(self, content, **kwargs): self.content = content
    class Text:
        @staticmethod
        def from_markup(text): return text
    class Align:
        @staticmethod
        def center(text, **kwargs): return text
    class Rule:
        def __init__(self, **kwargs): pass
    class Table:
        def __init__(self, *args, **kwargs): pass
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass

console = Console()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

for directory in [DATA_DIR, BACKUP_DIR, RESULTS_DIR, ML_MODELS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

CUT_ANK = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

PANA_SET = {
    1: ['119', '128', '137', '146', '155', '227', '236', '245', '290', '335', '344', '490', '580', '670', '390', '480', '570', '699', '799', '889', '590', '680'],
    2: ['110', '129', '138', '147', '156', '228', '237', '246', '255', '336', '345', '499', '589', '679', '789', '390', '480', '570', '688', '778'],
    3: ['111', '120', '139', '148', '157', '166', '229', '238', '247', '256', '337', '346', '355', '490', '580', '670', '788', '799'],
    4: ['112', '130', '149', '158', '167', '220', '239', '248', '257', '266', '338', '347', '356', '446', '590', '680', '789'],
    5: ['113', '122', '140', '159', '168', '177', '230', '249', '258', '267', '339', '348', '357', '447', '456', '690', '780'],
    6: ['114', '123', '150', '169', '178', '240', '259', '268', '277', '330', '349', '358', '367', '448', '457', '556'],
    7: ['115', '133', '124', '160', '179', '188', '223', '250', '269', '278', '340', '359', '368', '449', '458', '467', '557'],
    8: ['116', '125', '134', '170', '189', '224', '233', '260', '279', '288', '350', '369', '378', '440', '459', '468', '558', '567'],
    9: ['117', '126', '135', '144', '180', '199', '225', '234', '270', '289', '360', '379', '450', '469', '478', '559', '568', '667'],
    0: ['118', '127', '136', '190', '226', '235', '244', '299', '370', '389', '460', '479', '488', '550', '569', '578', '668', '776']
}

# --- ML PATTERN RECOGNITION ENGINE ---
class MLPatternEngine:
    """Advanced Machine Learning based pattern recognition with enhanced error handling"""
    
    def __init__(self, market_name):
        self.market_name = market_name
        self.model_file = os.path.join(ML_MODELS_DIR, f"{market_name}_model.pkl")
        self.patterns = defaultdict(list)
        self.load_model()
    
    def load_model(self):
        """Load saved ML model if exists"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.patterns = pickle.load(f)
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Model load failed - {e}[/yellow]")
            self.patterns = defaultdict(list)
    
    def save_model(self):
        """Save ML model for future use"""
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(dict(self.patterns), f)
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Model save failed - {e}[/yellow]")
    
    def train(self, df):
        """Train ML model on historical data with validation"""
        if len(df) < 10:
            return
        
        for i in range(len(df) - 1):
            try:
                curr = df.iloc[i]
                next_val = df.iloc[i + 1]
                
                if pd.isna(curr['close']) or pd.isna(next_val['open']) or pd.isna(curr['Jodi']):
                    continue
                
                key = f"close_to_open_{int(curr['close'])}"
                self.patterns[key].append(int(next_val['open']))
                
                jodi_key = f"jodi_{int(curr['Jodi'])}"
                self.patterns[jodi_key].append(int(next_val['Jodi']))
                
                day = curr['Date'].weekday()
                day_key = f"day_{day}_open"
                self.patterns[day_key].append(int(next_val['open']))
                
                sum_key = f"sum_{(int(curr['open']) + int(curr['close'])) % 10}"
                self.patterns[sum_key].append(int(next_val['open']))
            except (ValueError, TypeError, KeyError):
                continue
        
        self.save_model()
    
    def predict_with_confidence(self, last_record):
        """Predict next numbers with confidence score"""
        predictions = Counter()
        confidence_scores = {}
        
        try:
            if pd.isna(last_record['Jodi']) or pd.isna(last_record['close']):
                return [], {}
            
            close_key = f"close_to_open_{int(last_record['close'])}"
            if close_key in self.patterns:
                for val in self.patterns[close_key]:
                    predictions[val] += 2
            
            jodi_key = f"jodi_{int(last_record['Jodi'])}"
            if jodi_key in self.patterns:
                for jodi in self.patterns[jodi_key]:
                    try:
                        predictions[int(str(jodi).zfill(2)[0])] += 1
                    except (ValueError, IndexError):
                        pass
            
            day = last_record['Date'].weekday()
            day_key = f"day_{day}_open"
            if day_key in self.patterns:
                for val in self.patterns[day_key]:
                    predictions[val] += 1
            
            sum_key = f"sum_{(int(last_record['open']) + int(last_record['close'])) % 10}"
            if sum_key in self.patterns:
                for val in self.patterns[sum_key]:
                    predictions[val] += 1
            
            if predictions:
                total = sum(predictions.values())
                for num, count in predictions.items():
                    confidence_scores[num] = (count / total) * 100
            
            top_predictions = [num for num, _ in predictions.most_common(4)]
            return top_predictions, confidence_scores
        except Exception as e:
            console.print(f"[yellow]⚠️ ML Prediction error: {e}[/yellow]")
            return [], {}


# --- ADVANCED ANALYTICS ENGINE ---
class AdvancedAnalytics:
    """Advanced analytics and visualization"""
    
    @staticmethod
    def hot_cold_analysis(df, window=30):
        """Analyze hot and cold numbers"""
        try:
            recent = df.tail(window)
            all_nums = []
            for val in recent['open']:
                if not pd.isna(val):
                    all_nums.append(int(val))
            for val in recent['close']:
                if not pd.isna(val):
                    all_nums.append(int(val))
            
            counts = Counter(all_nums)
            hot_nums = [num for num, count in counts.most_common(3)]
            cold_nums = [num for num in range(10) if counts[num] <= 1]
            
            return {
                'hot_numbers': hot_nums if hot_nums else [0],
                'cold_numbers': cold_nums if cold_nums else [9],
                'frequency': dict(counts)
            }
        except Exception as e:
            console.print(f"[yellow]⚠️ Hot/Cold analysis error: {e}[/yellow]")
            return {'hot_numbers': [0], 'cold_numbers': [9], 'frequency': {}}
    
    @staticmethod
    def streak_analysis(df):
        """Analyze winning/losing streaks"""
        try:
            if len(df) < 2:
                return {'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0, 'streak_type': 'N/A'}
            
            streaks = []
            current = 0
            
            for i in range(1, len(df)):
                hist = df.iloc[:i]
                actual = df.iloc[i]
                
                if pd.isna(actual['open']) or pd.isna(actual['close']):
                    continue
                
                pred = find_brahmanda_sutra(hist)
                if not pred:
                    continue
                
                hit = int(actual['open']) in pred['core_otc'] or int(actual['close']) in pred['core_otc']
                
                if hit:
                    current = current + 1 if current > 0 else 1
                else:
                    current = current - 1 if current < 0 else -1
                
                streaks.append(current)
            
            max_win = max([s for s in streaks if s > 0], default=0)
            max_loss = abs(min([s for s in streaks if s < 0], default=0))
            current_streak = streaks[-1] if streaks else 0
            
            return {
                'current_streak': abs(current_streak),
                'streak_type': 'WIN' if current_streak > 0 else 'LOSS' if current_streak < 0 else 'NEUTRAL',
                'max_win_streak': max_win,
                'max_loss_streak': max_loss
            }
        except Exception as e:
            console.print(f"[yellow]⚠️ Streak analysis error: {e}[/yellow]")
            return {'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0, 'streak_type': 'N/A'}
    
    @staticmethod
    def best_time_analysis(df):
        """Find best performing days"""
        try:
            if len(df) < 20:
                return None
            
            day_performance = defaultdict(lambda: {'total': 0, 'wins': 0})
            
            for i in range(1, len(df)):
                hist = df.iloc[:i]
                actual = df.iloc[i]
                
                if pd.isna(actual['open']) or pd.isna(actual['close']):
                    continue
                
                pred = find_brahmanda_sutra(hist)
                if not pred:
                    continue
                
                day = actual['Date'].weekday()
                day_performance[day]['total'] += 1
                
                if int(actual['open']) in pred['core_otc'] or int(actual['close']) in pred['core_otc']:
                    day_performance[day]['wins'] += 1
            
            if not day_performance:
                return None
            
            best_day = max(day_performance.items(), 
                           key=lambda x: x[1]['wins']/x[1]['total'] if x[1]['total'] > 0 else 0)
            
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            return {
                'best_day': days[best_day[0]],
                'win_rate': (best_day[1]['wins']/best_day[1]['total']*100) if best_day[1]['total'] > 0 else 0
            }
        except Exception as e:
            console.print(f"[yellow]⚠️ Best time analysis error: {e}[/yellow]")
            return None
    
    @staticmethod
    def create_heatmap_text(frequency):
        """Create text-based heatmap"""
        try:
            if not frequency:
                return "No data"
            
            max_freq = max(frequency.values()) if frequency else 1
            heatmap = []
            
            for num in range(10):
                count = frequency.get(num, 0)
                intensity = int((count / max_freq) * 5) if max_freq > 0 else 0
                colors = ['dim white', 'white', 'yellow', 'bold yellow', 'bold red', 'bold bright_red']
                color = colors[min(intensity, 5)]
                heatmap.append(f"[{color}]{num}:{count:02d}[/{color}]")
            
            return " | ".join(heatmap)
        except Exception:
            return "No data"


# --- AUTO BACKUP SYSTEM ---
class BackupSystem:
    """Automatic backup and restore system"""
    
    @staticmethod
    def create_backup(market_name, prediction_data):
        """Create backup of predictions"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(BACKUP_DIR, f"{market_name}_{timestamp}.json")
            with open(backup_file, 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
            return True
        except Exception as e:
            console.print(f"[yellow]⚠️ Backup failed: {e}[/yellow]")
            return False
    
    @staticmethod
    def save_result(market_name, result_data):
        """Save prediction results"""
        try:
            result_file = os.path.join(RESULTS_DIR, f"{market_name}_results.json")
            existing = []
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    existing = json.load(f)
            existing.append(result_data)
            with open(result_file, 'w') as f:
                json.dump(existing, f, indent=2, default=str)
            return True
        except Exception as e:
            console.print(f"[yellow]⚠️ Result save failed: {e}[/yellow]")
            return False


# --- EXPORT SYSTEM ---
class ExportSystem:
    """Export reports to various formats"""
    
    @staticmethod
    def export_to_text(market_name, data):
        """Export to text file"""
        try:
            filename = os.path.join(RESULTS_DIR, f"{market_name}_report_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"=== {market_name.upper()} - PREDICTION REPORT ===\n")
                f.write(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n\n")
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            console.print(f"[green]✅ Report exported: {filename}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
            return False


# --- ENHANCED DATA LOADER ---
def load_and_prepare_data(filepath):
    """Enhanced data loading with comprehensive error handling"""
    try:
        if not os.path.exists(filepath):
            console.print(f"[bold red]❌ File not found: {filepath}[/bold red]")
            return None
        
        df = pd.read_csv(filepath, sep=r'\s*/\s*', header=None, engine='python',
                         names=['Date_Str', 'Pana_Jodi_Pana'])
        
        df = df.dropna(subset=['Pana_Jodi_Pana'])
        df = df[~df['Pana_Jodi_Pana'].str.contains(r"\*|x", na=False, case=False)]
        df[['Open_Pana', 'Jodi', 'Close_Pana']] = df['Pana_Jodi_Pana'].str.split(r'\s*-\s*', expand=True)
        
        for col in ['Open_Pana', 'Jodi', 'Close_Pana']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Open_Pana', 'Jodi', 'Close_Pana'])
        df = df.astype({'Open_Pana': int, 'Jodi': int, 'Close_Pana': int})
        
        df['open'] = df['Jodi'].apply(lambda x: int(str(x).zfill(2)[0]))
        df['close'] = df['Jodi'].apply(lambda x: int(str(x).zfill(2)[1]))
        
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            for fmt in ['%d-%m-%Y', '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y']:
                try:
                    return datetime.strptime(str(date_str).strip(), fmt).date()
                except ValueError:
                    pass
            return pd.NaT
        
        df['Date'] = df['Date_Str'].apply(parse_date)
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        if len(df) > 0:
            console.print(f"[bold green]✔️ Data loaded: {len(df)} entries[/bold green]")
        else:
            console.print("[bold red]❌ No valid data found[/bold red]")
            return None
        
        return df
    except Exception as e:
        console.print(f"[bold red]❌ Data load error: {e}[/bold red]")
        return None


# --- CORE PREDICTION ENGINES ---
def find_brahmanda_sutra(df):
    """Optimized prediction engine with validation"""
    try:
        if len(df) < 1:
            return None
        
        last_record = df.iloc[-1]
        
        if pd.isna(last_record['close']) or pd.isna(last_record['open']):
            return None
        
        close_ank = int(last_record['close'])
        cut_ank = CUT_ANK[close_ank]
        jodi_sum = (int(last_record['open']) + int(last_record['close'])) % 10
        sum_cut = CUT_ANK[jodi_sum]
        
        core_otc = sorted(list(set([close_ank, cut_ank, jodi_sum, sum_cut])))
        jodis = [f"{a}{b}" for a in core_otc for b in core_otc]
        
        return {
            'core_otc': core_otc,
            'strongest_jodi': jodis
        }
    except Exception as e:
        console.print(f"[yellow]⚠️ Brahmanda sutra error: {e}[/yellow]")
        return None


def find_jodi_seed_trend(df):
    """Optimized trend analysis"""
    try:
        if len(df) < 5:
            return {'top_anuman_ank': []}
        
        recent_df = df.tail(40)
        all_anks = []
        for val in recent_df['open']:
            if not pd.isna(val):
                all_anks.append(int(val))
        for val in recent_df['close']:
            if not pd.isna(val):
                all_anks.append(int(val))
        
        ank_counts = Counter(all_anks)
        top_anks = [num for num, _ in ank_counts.most_common(4)]
        
        return {
            'top_anuman_ank': sorted(top_anks) if top_anks else [0]
        }
    except Exception as e:
        console.print(f"[yellow]⚠️ Trend analysis error: {e}[/yellow]")
        return {'top_anuman_ank': [0]}


def get_suggested_panels(core_otc):
    """Optimized panel suggestions"""
    try:
        panels = []
        for ank in core_otc:
            if ank in PANA_SET and PANA_SET[ank]:
                panels.extend(random.sample(PANA_SET[ank], min(len(PANA_SET[ank]), 2)))
        return sorted(list(set(panels)))[:8]
    except Exception:
        return []


# --- VALIDATION ---
def validate_predictions(df):
    """Validate predictions with comprehensive error handling"""
    try:
        if len(df) < 2:
            return {'total_days': 0, 'ank_hit_count': 0, 'all_results': []}, {}
        
        df_validate = df.tail(41).copy()
        results = []
        ank_hit_count = 0
        
        for i in range(1, len(df_validate)):
            try:
                historical_df = df_validate.iloc[:i]
                actual_record = df_validate.iloc[i]
                
                if pd.isna(actual_record['Open_Pana']) or pd.isna(actual_record['Close_Pana']):
                    continue
                if pd.isna(actual_record['open']) or pd.isna(actual_record['close']):
                    continue
                
                prediction = find_brahmanda_sutra(historical_df)
                if not prediction:
                    continue
                
                otc = prediction['core_otc']
                
                ank_hits = []
                if int(actual_record['open']) in otc:
                    ank_hits.append(str(int(actual_record['open'])))
                if int(actual_record['close']) in otc:
                    ank_hits.append(str(int(actual_record['close'])))
                
                if ank_hits:
                    ank_hit_count += 1
                    unique_ank_hits = sorted(list(set(ank_hits)))
                    ank_display = f"{'/'.join(unique_ank_hits)}"
                    ank_hit_raw = "PASS"
                else:
                    ank_display = "FAIL"
                    ank_hit_raw = "FAIL"
                
                try:
                    actual_open_pana_sum = sum(int(d) for d in str(int(actual_record['Open_Pana']))) % 10
                    actual_close_pana_sum = sum(int(d) for d in str(int(actual_record['Close_Pana']))) % 10
                except (ValueError, TypeError):
                    continue
                
                pana_hits = []
                if actual_open_pana_sum in otc:
                    pana_hits.append(str(int(actual_record['Open_Pana'])))
                if actual_close_pana_sum in otc:
                    pana_hits.append(str(int(actual_record['Close_Pana'])))
                
                pana_display = f"{'/'.join(pana_hits)}" if pana_hits else "FAIL"
                
                results.append({
                    'date': actual_record['Date'],
                    'open_pana': int(actual_record['Open_Pana']),
                    'close_pana': int(actual_record['Close_Pana']),
                    'jodi': int(actual_record['Jodi']),
                    'prediction_otc': ' '.join(map(str, otc)),
                    'ank_status': ank_display,
                    'pana_status': pana_display,
                    'ank_hit_raw': ank_hit_raw
                })
            except Exception:
                continue
        
        yesterday_validation = results[-1] if results else {}
        
        return {
            'total_days': len(results),
            'ank_hit_count': ank_hit_count,
            'all_results': results
        }, yesterday_validation
    except Exception as e:
        console.print(f"[yellow]⚠️ Validation error: {e}[/yellow]")
        return {'total_days': 0, 'ank_hit_count': 0, 'all_results': []}, {}


def track_weekly_performance(all_results):
    """Track last week performance"""
    try:
        if not all_results:
            return []
        
        weekly_data = all_results[-7:]
        scoreboard = []
        for entry in weekly_data:
            ank_display = f"[bold bright_green]{entry['ank_status']}[/bold bright_green]" if entry['ank_status'] != 'FAIL' else "[bold red]FAIL[/bold red]"
            pana_display = f"[bold bright_green]{entry['pana_status']}[/bold bright_green]" if entry['pana_status'] != 'FAIL' else "[bold red]FAIL[/bold red]"
            
            scoreboard.append({
                'date_day': entry['date'].strftime('%d-%b (%a)'),
                'jodi': str(entry['jodi']).zfill(2),
                'ank_status': ank_display,
                'pana_status': pana_display
            })
        return scoreboard
    except Exception:
        return []


# --- MULTI-MARKET COMPARISON ---
def compare_markets():
    """Compare all available markets"""
    try:
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
        
        if not files:
            console.print("[yellow]No markets to compare[/yellow]")
            return
        
        comparison_table = Table(title="🔥 MULTI-MARKET COMPARISON", border_style="cyan", show_header=True)
        comparison_table.add_column("Market", style="yellow")
        comparison_table.add_column("Success Rate", justify="center")
        comparison_table.add_column("Best Day", justify="center")
        comparison_table.add_column("Status", justify="center")
        
        for file in files:
            market_name = file.replace('.txt', '')
            filepath = os.path.join(DATA_DIR, file)
            df = load_and_prepare_data(filepath)
            
            if df is None or len(df) < 5:
                continue
            
            val_results, _ = validate_predictions(df)
            if val_results['total_days'] > 0:
                success_rate = (val_results['ank_hit_count'] / val_results['total_days']) * 100
                analytics = AdvancedAnalytics()
                best_time = analytics.best_time_analysis(df)
                best_day = best_time['best_day'] if best_time else 'N/A'
                status = "🔥 HOT" if success_rate >= 70 else "⚡ GOOD" if success_rate >= 50 else "❄️ COLD"
                
                comparison_table.add_row(market_name.upper(), f"{success_rate:.1f}%", best_day, status)
        
        console.print(comparison_table)
        console.input("\n[bold white]Press ENTER to continue...[/bold white]")
    except Exception as e:
        console.print(f"[red]❌ Comparison error: {e}[/red]")


# --- MARKET SELECTION ---
def get_market_name_interactively():
    """Enhanced market selection with stats"""
    try:
        if not os.path.exists(DATA_DIR):
            console.print(f"[bold red]❌ Data directory not found[/bold red]")
            sys.exit(1)
        
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
        market_names = sorted([f.replace('.txt', '') for f in files])
        
        if not market_names:
            console.print("[bold red]❌ No market files found[/bold red]")
            sys.exit(1)
        
        console.print(Rule(style="bold yellow", title="🎯 BAZAAR CHUNAV (MARKET SELECTION)"))
        
        if RICH_AVAILABLE:
            table = Table(title="📊 Available Markets", border_style="magenta", show_header=True)
            table.add_column("No.", style="cyan", justify="center")
            table.add_column("Market Name", style="green")
            table.add_column("Status", justify="center")
            
            for i, name in enumerate(market_names, 1):
                filepath = os.path.join(DATA_DIR, f"{name}.txt")
                df = load_and_prepare_data(filepath)
                
                if df is not None and len(df) >= 5:
                    val_results, _ = validate_predictions(df)
                    if val_results['total_days'] > 0:
                        rate = (val_results['ank_hit_count'] / val_results['total_days']) * 100
                        status = f"[green]{rate:.0f}%[/green]" if rate >= 60 else f"[yellow]{rate:.0f}%[/yellow]"
                    else:
                        status = "[dim]N/A[/dim]"
                else:
                    status = "[dim]N/A[/dim]"
                
                table.add_row(str(i), name.upper(), status)
            
            console.print(table)
        
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("[white]  • Enter number to select market[/white]")
        console.print("[white]  • Press [bold]C[/bold] for Market Comparison[/white]")
        console.print("[white]  • Press [bold]0[/bold] to Exit[/white]")
        
        while True:
            try:
                choice = console.input("\n[bold white]Your choice: [/bold white]").strip().upper()
                
                if choice == '0':
                    return None
                elif choice == 'C':
                    compare_markets()
                    os.system('cls' if os.name == 'nt' else 'clear')
                    return get_market_name_interactively()
                
                index = int(choice) - 1
                if 0 <= index < len(market_names):
                    return market_names[index]
                else:
                    console.print("[bold red]❌ Invalid number[/bold red]")
            except ValueError:
                console.print("[bold red]❌ Enter valid option[/bold red]")
            except (EOFError, KeyboardInterrupt):
                return None
    except Exception as e:
        console.print(f"[red]❌ Market selection error: {e}[/red]")
        return None


# --- ENHANCED DISPLAY ---
def display_final_output(market_name, sutra_analysis, last_record, validation_results,
                         yesterday_validation, weekly_performance, trend_analysis,
                         ml_predictions, ml_confidence, hot_cold, streak_info, best_time):
    """Ultimate display with all new features"""
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if not RICH_AVAILABLE:
            print(f"--- {market_name.upper()} - PROJECT PARAMAANU v5.0 PRO MAX ---")
            return "0"
        
        prediction_date = last_record['Date'] + timedelta(days=1)
        prediction_date_str = prediction_date.strftime('%d-%m-%Y')
        
        console.print(Panel(
            Align.center(Text.from_markup(f'🔱 {market_name.upper()} - PARAMAANU v5.0 PRO MAX 🔱\n[dim]AI-Powered Ultimate Edition[/dim]', justify="center")),
            border_style="magenta",
            padding=(1, 2)
        ))
        
        sutra_set = set(sutra_analysis.get('core_otc', []))
        trend_set = set(trend_analysis.get('top_anuman_ank', []))
        ml_set = set(ml_predictions)
        
        ultra_master = sorted(list(sutra_set.intersection(trend_set).intersection(ml_set)))
        common_sutra_trend = sorted(list(sutra_set.intersection(trend_set)))
        
        confidence_text = ""
        if ml_confidence:
            for num in ml_predictions:
                conf = ml_confidence.get(num, 0)
                color = "bright_green" if conf > 50 else "yellow" if conf > 30 else "white"
                confidence_text += f"[{color}]{num}({conf:.0f}%)[/{color}] "
        
        grid1 = Table.grid(padding=(0, 2), expand=True)
        grid1.add_column()
        grid1.add_column()
        
        ai_panel = Text.from_markup(
            f"🎯 Date: [bold yellow]{prediction_date_str}[/bold yellow]\n"
            f"🔮 Brahmanda Sutra: [bold bright_yellow]{' '.join(map(str, sutra_analysis.get('core_otc', [])))}[/bold bright_yellow]\n"
            f"📊 Trend Analysis: [bold cyan]{' '.join(map(str, trend_set))}[/bold cyan]\n"
            f"🤖 ML Predictions: {confidence_text}\n"
            f"{'⚡ ULTRA MASTER KEY: [bold bright_red on white] ' + ' '.join(map(str, ultra_master)) + ' [/bold bright_red on white]' if ultra_master else '👑 MASTER KEY: [bold bright_red]' + ' '.join(map(str, common_sutra_trend)) + '[/bold bright_red]'}"
        )
        
        jodi_panel = Text.from_markup(
            f"🎯 TOP JODIS: [bold green]{' '.join(sutra_analysis.get('strongest_jodi', [])[:12])}[/bold green]\n"
            f"🎲 PANELS: [bold white]{' '.join(get_suggested_panels(sutra_analysis.get('core_otc', [])))}[/bold white]"
        )
        
        grid1.add_row(
            Panel(ai_panel, title="[red]🧠 AI BRAIN - DIVYA SANKET[/red]", border_style="red"),
            Panel(jodi_panel, title="[green]🎰 GAME PLAN[/green]", border_style="green")
        )
        console.print(grid1)
        
        grid2 = Table.grid(padding=(0, 2), expand=True)
        grid2.add_column()
        grid2.add_column()
        
        hot_cold_text = Text.from_markup(
            f"🔥 HOT NUMBERS: [bold red]{' '.join(map(str, hot_cold['hot_numbers']))}[/bold red]\n"
            f"❄️ COLD NUMBERS: [bold blue]{' '.join(map(str, hot_cold['cold_numbers']))}[/bold blue]\n"
            f"📊 FREQUENCY MAP:\n{AdvancedAnalytics.create_heatmap_text(hot_cold['frequency'])}"
        )
        
        best_time_text = ''
        if best_time:
            win_rate = best_time['win_rate']
            best_time_text = f"📅 Best Day: [bold yellow]{best_time['best_day']} ({win_rate:.0f}%)[/bold yellow]"
        else:
            best_time_text = '[dim]Calculating...[/dim]'
        
        streak_color = "green" if streak_info['streak_type'] == 'WIN' else "red" if streak_info['streak_type'] == 'LOSS' else "yellow"
        streak_text = Text.from_markup(
            f"Current Streak: [{streak_color}]{streak_info['streak_type']} ({streak_info['current_streak']})[/{streak_color}]\n"
            f"🏆 Max Win Streak: [green]{streak_info['max_win_streak']}[/green]\n"
            f"💔 Max Loss Streak: [red]{streak_info['max_loss_streak']}[/red]\n"
            f"{best_time_text}"
        )
        
        grid2.add_row(
            Panel(hot_cold_text, title="[yellow]🔥 HOT/COLD ANALYSIS[/yellow]", border_style="yellow"),
            Panel(streak_text, title="[cyan]📈 STREAK & TIMING[/cyan]", border_style="cyan")
        )
        console.print(grid2)
        
        if yesterday_validation:
            yesterday_panel = Text.from_markup(
                f"📅 Date: {yesterday_validation['date'].strftime('%d-%b-%Y')}\n"
                f"Result: [magenta]{yesterday_validation['open_pana']}-{str(yesterday_validation['jodi']).zfill(2)}-{yesterday_validation['close_pana']}[/magenta]\n"
                f"Our Prediction: [yellow]{yesterday_validation['prediction_otc']}[/yellow]\n"
                f"Ank: {yesterday_validation['ank_status']} | Pana: {yesterday_validation['pana_status']}"
            )
            console.print(Panel(yesterday_panel, title="[cyan]📋 YESTERDAY'S REPORT[/cyan]", border_style="cyan"))
        
        if weekly_performance:
            score_table = Table(border_style="yellow", show_header=True, header_style="bold yellow")
            score_table.add_column("Date", style="cyan")
            score_table.add_column("Jodi", justify="center")
            score_table.add_column("Ank", justify="center")
            score_table.add_column("Pana", justify="center")
            
            for entry in weekly_performance:
                ank_display = f"[bold bright_green]{entry['ank_status']}[/bold bright_green]" if entry['ank_status'] != 'FAIL' else "[bold red]FAIL[/bold red]"
                pana_display = f"[bold bright_green]{entry['pana_status']}[/bold bright_green]" if entry['pana_status'] != 'FAIL' else "[bold red]FAIL[/bold red]"
                score_table.add_row(entry['date_day'], entry['jodi'], ank_display, pana_display)
            
            console.print(Panel(score_table, title="[yellow]🏆 WEEKLY SCOREBOARD[/yellow]", border_style="yellow"))
        
        if validation_results['total_days'] > 0:
            ank_rate = (validation_results['ank_hit_count'] / validation_results['total_days']) * 100
            summary_color = "green" if ank_rate >= 70 else "yellow" if ank_rate >= 50 else "red"
            
            summary_text = Text.from_markup(
                f"📊 LAST 40 DAYS PERFORMANCE\n"
                f"Total Games: [cyan]{validation_results['total_days']}[/cyan]\n"
                f"Ank Hits: [green]{validation_results['ank_hit_count']}[/green]\n"
                f"🎯 SUCCESS RATE: [{summary_color}]{ank_rate:.2f}%[/{summary_color}]"
            )
            console.print(Panel(Align.center(summary_text), title="[magenta]📈 HISTORICAL POWER[/magenta]", border_style="magenta"))
        
        console.print(Rule(style="dim white"))
        console.print(Align.center(f"[bold magenta]{market_name.upper()}[/bold magenta] | [bold cyan]AI-Powered by AANU v5.0[/bold cyan]"))
        console.print(Align.center("[dim]Created by S-T & Aanu-AI Reborn[/dim]"))
        
        console.print("\n[bold cyan]OPTIONS:[/bold cyan]")
        console.print("[white]  [ENTER] Main Menu  |  [E] Export Report  |  [0] Exit[/white]")
        
        return console.input("\n[bold white]Your choice: [/bold white]").strip().upper()
    except Exception as e:
        console.print(f"[red]❌ Display error: {e}[/red]")
        return "0"


# --- MAIN ANALYSIS ENGINE ---
def run_analysis(market_name):
    """Enhanced analysis with all new features"""
    try:
        data_file = os.path.join(DATA_DIR, f"{market_name}.txt")
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Loading data...", total=None)
                df = load_and_prepare_data(data_file)
                progress.update(task, completed=True)
        else:
            df = load_and_prepare_data(data_file)
        
        if df is None or df.empty:
            return console.input("[bold red]❌ Data load failed. Press ENTER...[/bold red]").strip()
        
        if len(df) < 2:
            console.print("[bold red]❌ Insufficient data (need at least 2 days)[/bold red]")
            return console.input("[bold white]Press ENTER...[/bold white]").strip()
        
        last_record = df.iloc[-1].to_dict()
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Training AI Brain...", total=None)
                ml_engine = MLPatternEngine(market_name)
                ml_engine.train(df)
                ml_predictions, ml_confidence = ml_engine.predict_with_confidence(last_record)
                progress.update(task, completed=True)
                
                task = progress.add_task("[cyan]Running predictions...", total=None)
                sutra_analysis = find_brahmanda_sutra(df)
                trend_analysis = find_jodi_seed_trend(df)
                progress.update(task, completed=True)
                
                task = progress.add_task("[cyan]Analyzing patterns...", total=None)
                analytics = AdvancedAnalytics()
                hot_cold = analytics.hot_cold_analysis(df)
                streak_info = analytics.streak_analysis(df)
                best_time = analytics.best_time_analysis(df)
                progress.update(task, completed=True)
                
                task = progress.add_task("[cyan]Validating accuracy...", total=None)
                validation_results, yesterday_validation = validate_predictions(df)
                weekly_performance = track_weekly_performance(validation_results.get('all_results', []))
                progress.update(task, completed=True)
        else:
            ml_engine = MLPatternEngine(market_name)
            ml_engine.train(df)
            ml_predictions, ml_confidence = ml_engine.predict_with_confidence(last_record)
            sutra_analysis = find_brahmanda_sutra(df)
            trend_analysis = find_jodi_seed_trend(df)
            analytics = AdvancedAnalytics()
            hot_cold = analytics.hot_cold_analysis(df)
            streak_info = analytics.streak_analysis(df)
            best_time = analytics.best_time_analysis(df)
            validation_results, yesterday_validation = validate_predictions(df)
            weekly_performance = track_weekly_performance(validation_results.get('all_results', []))
        
        backup_data = {
            'market': market_name,
            'date': str(last_record['Date'] + timedelta(days=1)),
            'predictions': {
                'sutra': sutra_analysis.get('core_otc', []),
                'trend': trend_analysis.get('top_anuman_ank', []),
                'ml': ml_predictions,
                'confidence': ml_confidence
            }
        }
        BackupSystem.create_backup(market_name, backup_data)
        
        user_choice = display_final_output(
            market_name=market_name,
            sutra_analysis=sutra_analysis,
            last_record=last_record,
            validation_results=validation_results,
            yesterday_validation=yesterday_validation,
            weekly_performance=weekly_performance,
            trend_analysis=trend_analysis,
            ml_predictions=ml_predictions,
            ml_confidence=ml_confidence,
            hot_cold=hot_cold,
            streak_info=streak_info,
            best_time=best_time
        )
        
        if user_choice == 'E':
            export_data = {
                'Market': market_name.upper(),
                'Prediction Date': str(last_record['Date'] + timedelta(days=1)),
                'Brahmanda Sutra': ' '.join(map(str, sutra_analysis.get('core_otc', []))),
                'ML Predictions': ' '.join(map(str, ml_predictions)),
                'Hot Numbers': ' '.join(map(str, hot_cold['hot_numbers'])),
                'Success Rate': f"{(validation_results['ank_hit_count'] / validation_results['total_days'] * 100):.2f}%" if validation_results['total_days'] > 0 else 'N/A'
            }
            ExportSystem.export_to_text(market_name, export_data)
            return console.input("\n[bold white]Press ENTER to continue...[/bold white]").strip()
        
        return user_choice
        
    except Exception as e:
        console.print(f"[bold red]❌ Analysis error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return console.input("[bold white]Press ENTER...[/bold white]").strip()


# --- MAIN FUNCTION ---
def main():
    """Main execution loop"""
    if not RICH_AVAILABLE:
        print("⚠️ Warning: 'rich' library not installed. Limited features.")
        print("Install: pip install rich pandas numpy")
    
    if len(sys.argv) > 1:
        if run_analysis(sys.argv[1].strip()) == '0':
            sys.exit(0)
        sys.exit(0)
    
    if RICH_AVAILABLE:
        os.system('cls' if os.name == 'nt' else 'clear')
        welcome = Panel(
            Align.center(Text.from_markup(
                "[bold yellow]🔱 PARAMAANU v5.0 PRO MAX 🔱[/bold yellow]\n\n"
                "[cyan]AI-Powered Ultimate Edition[/cyan]\n\n"
                "[green]✨ NEW FEATURES:[/green]\n"
                "[white]• Machine Learning Pattern Recognition[/white]\n"
                "[white]• Advanced Analytics & Heatmaps[/white]\n"
                "[white]• Hot/Cold Number Analysis[/white]\n"
                "[white]• Streak Tracking System[/white]\n"
                "[white]• Auto-Backup & Export[/white]\n"
                "[white]• Multi-Market Comparison[/white]\n"
                "[white]• 100% Error Handling[/white]\n\n"
                "[dim]Created by S-T & Aanu-AI Reborn[/dim]"
            )),
            border_style="magenta",
            padding=(2, 4)
        )
        console.print(welcome)
        console.input("\n[bold white]Press ENTER to start...[/bold white]")
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        market_name = get_market_name_interactively()
        
        if market_name is None:
            console.print("\n[bold cyan]👋 Dhanyavaad! Goodbye...[/bold cyan]")
            break
        
        user_input = run_analysis(market_name)
        
        if user_input == '0':
            console.print("\n[bold cyan]👋 Dhanyavaad! Goodbye...[/bold cyan]")
            break


if __name__ == "__main__":
    main()
