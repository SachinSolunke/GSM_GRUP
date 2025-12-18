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
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def __init__(self): pass
        def print(self, *args, **kwargs): print(*args)
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
    1: ['119', '128', '137', '146', '155', '227', '236', '245'], 2: ['110', '129', '138', '147', '156', '228', '237', '246'],
    3: ['111', '120', '139', '148', '157', '166', '229', '238'], 4: ['112', '130', '149', '158', '167', '220', '239', '248'],
    5: ['113', '122', '140', '159', '168', '177', '230', '249'], 6: ['114', '123', '150', '169', '178', '240', '259', '268'],
    7: ['115', '133', '124', '160', '179', '188', '223', '250'], 8: ['116', '125', '134', '170', '189', '224', '233', '260'],
    9: ['117', '126', '135', '144', '180', '199', '225', '234'], 0: ['118', '127', '136', '190', '226', '235', '244', '299']
}

# ==================== ACCURACY BOOSTING ENGINES ====================

class MLPatternEngine:
    def __init__(self, market_name):
        self.market_name = market_name
        self.model_file = os.path.join(ML_MODELS_DIR, f"{market_name}_model.pkl")
        self.patterns = defaultdict(list)
        self.load_model()
    
    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.patterns = pickle.load(f)
            except: 
                self.patterns = defaultdict(list)
    
    def save_model(self):
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(dict(self.patterns), f)
        except: 
            pass
    
    def train(self, df):
        if len(df) < 10: 
            return
        
        for i in range(len(df) - 1):
            try:
                curr = df.iloc[i]
                next_val = df.iloc[i + 1]
                
                # Check kaaro ke data available hai
                if pd.notna(curr['close']) and pd.notna(next_val['open']):
                    self.patterns[f"close_{int(curr['close'])}"].append(int(next_val['open']))
                
                if pd.notna(curr['Jodi']) and pd.notna(next_val['Jodi']):
                    self.patterns[f"jodi_{int(curr['Jodi'])}"].append(int(next_val['Jodi']))
                
                if pd.notna(curr['Date']) and pd.notna(next_val['open']):
                    weekday = curr['Date'].weekday()
                    self.patterns[f"day_{weekday}"].append(int(next_val['open']))
            
            except (ValueError, KeyError, TypeError):
                continue
        
        self.save_model()
    
    def predict_with_confidence(self, last_record):
        predictions = Counter()
        
        try:
            close_val = int(last_record['close'])
            key = f"close_{close_val}"
            
            if key in self.patterns and self.patterns[key]:
                for val in self.patterns[key]:
                    predictions[val] += 2
        except:
            pass
        
        top_predictions = [num for num, _ in predictions.most_common(4)]
        confidence_scores = {num: 60 for num in top_predictions}
        
        return top_predictions, confidence_scores


class HotRepeatPattern:
    def analyze_hot_repeats(self, df):
        if len(df) < 5: 
            return {}
        
        recent = df.tail(30)
        repeat_patterns = defaultdict(list)
        
        for i in range(len(recent)):
            try:
                current = recent.iloc[i]
                for j in range(max(0, i-5), i):
                    past = recent.iloc[j]
                    days_diff = (current['Date'] - past['Date']).days
                    if 1 <= days_diff <= 5:
                        if current['open'] == past['open']:
                            repeat_patterns[current['open']].append(days_diff)
            except:
                continue
        
        return sorted([(num, len(dates)) for num, dates in repeat_patterns.items()], key=lambda x: x[1], reverse=True)
    
    def get_top_repeats(self, df, limit=4):
        repeats = self.analyze_hot_repeats(df)
        return [num for num, _ in repeats[:limit]]


class WeeklyCycleDetector:
    @staticmethod
    def predict_today(df, today_weekday):
        if len(df) < 20: 
            return []
        
        day_data = df[df['Date'].dt.weekday == today_weekday]
        if len(day_data) < 2: 
            return []
        
        opens = day_data['open'].tail(4).values
        return sorted(list(set(opens)))


class JodiReverseMapper:
    @staticmethod
    def predict_jodi_from_pattern(df):
        if len(df) < 5: 
            return []
        
        recent = df.tail(10)
        jodi_counter = Counter(recent['Jodi'])
        return sorted([j for j, c in jodi_counter.items() if c >= 1])[:6]


class HistoricalExactMatch:
    @staticmethod
    def get_prediction_from_matches(df, window=3):
        if len(df) < window + 5: 
            return {'from_matches': False, 'confidence': 0, 'likely_opens': [], 'likely_closes': []}
        
        try:
            recent_seq = tuple(df.tail(window)['Jodi'].values)
            matches = []
            
            for i in range(len(df) - window - 1):
                if tuple(df.iloc[i:i+window]['Jodi'].values) == recent_seq and i + window < len(df):
                    matches.append({
                        'open': df.iloc[i + window]['open'],
                        'close': df.iloc[i + window]['close']
                    })
            
            if matches:
                opens = Counter([m['open'] for m in matches])
                closes = Counter([m['close'] for m in matches])
                return {
                    'from_matches': True, 
                    'confidence': len(matches), 
                    'likely_opens': [o for o, _ in opens.most_common(2)], 
                    'likely_closes': [c for c, _ in closes.most_common(2)]
                }
        except:
            pass
        
        return {'from_matches': False, 'confidence': 0, 'likely_opens': [], 'likely_closes': []}


def combine_all_predictions(df, ml_predictions=None):
    if len(df) < 5: 
        return None
    
    last_record = df.iloc[-1].to_dict()
    today_weekday = last_record['Date'].weekday()
    
    all_predictions = {'sources': {}, 'combined': [], 'confidence_score': {}}
    
    try:
        hot_repeat = HotRepeatPattern()
        all_predictions['sources']['hot_repeats'] = hot_repeat.get_top_repeats(df, limit=4)
    except: 
        all_predictions['sources']['hot_repeats'] = []
    
    try:
        all_predictions['sources']['weekly_cycle'] = WeeklyCycleDetector.predict_today(df, today_weekday)
    except: 
        all_predictions['sources']['weekly_cycle'] = []
    
    try:
        jodi_pred = JodiReverseMapper.predict_jodi_from_pattern(df)
        all_predictions['sources']['jodi_mapper'] = [int(str(j).zfill(2)[0]) for j in jodi_pred] + [int(str(j).zfill(2)[1]) for j in jodi_pred]
    except: 
        all_predictions['sources']['jodi_mapper'] = []
    
    try:
        match_pred = HistoricalExactMatch.get_prediction_from_matches(df, window=3)
        if match_pred['from_matches']:
            all_predictions['sources']['exact_match'] = match_pred['likely_opens'] + match_pred['likely_closes']
            all_predictions['exact_match_confidence'] = match_pred['confidence']
    except: 
        all_predictions['sources']['exact_match'] = []
    
    combined_counter = Counter()
    for source_name, predictions in all_predictions['sources'].items():
        for pred in predictions:
            if pred is not None:
                weight = {'exact_match': 3, 'hot_repeats': 2, 'weekly_cycle': 2, 'jodi_mapper': 1}.get(source_name, 1)
                combined_counter[pred] += weight
    
    if combined_counter:
        final_predictions = [num for num, _ in combined_counter.most_common(4)]
        total_weight = sum(combined_counter.values())
        for num in final_predictions:
            all_predictions['confidence_score'][num] = (combined_counter[num] / total_weight) * 100
        all_predictions['combined'] = final_predictions
    
    return all_predictions

def get_super_master_key(df, sutra_analysis, combined_predictions):
    sutra_set = set(sutra_analysis.get('core_otc', []))
    combined_set = set(combined_predictions.get('combined', []))
    super_key = sorted(list(sutra_set.intersection(combined_set)))
    return super_key if super_key else combined_predictions.get('combined', [])

# ==================== ADVANCED FEATURES ====================

class RealtimePredictionDashboard:
    def __init__(self, market_name):
        self.market_name = market_name
    
    def calculate_confidence_meter(self, prediction_data):
        if not prediction_data: return 0
        methods = sum([1 for key in ['sutra', 'trend', 'ml', 'combined'] if prediction_data.get(key)])
        return (methods / 4) * 100
    
    def create_confidence_bar(self, confidence):
        bars = int(confidence / 10)
        color = "[bold bright_green]" if confidence >= 80 else "[bold yellow]" if confidence >= 40 else "[bold red]"
        return f"{color}{'█' * bars}{'░' * (10-bars)}[/] {confidence:.0f}%"
    
    def get_prediction_strength(self, predictions_dict):
        if not predictions_dict: return "WEAK ⚠️"
        score = 40 if predictions_dict.get('super_key') else 0
        score += 30 if len(predictions_dict.get('sutra', [])) >= 3 else 20 if len(predictions_dict.get('sutra', [])) >= 2 else 10
        if strength_score >= 80: return "ULTRA STRONG 🔥🔥🔥"
        elif score >= 60: return "VERY STRONG 🔥🔥"
        elif score >= 40: return "STRONG 🔥"
        return "MODERATE ⚡"


class PatternHistoryTracker:
    def __init__(self, market_name):
        self.market_name = market_name
        self.pattern_history = defaultdict(lambda: {'hits': 0, 'attempts': 0})
    
    def get_success_rates(self):
        return {p: {'rate': (d['hits']/d['attempts']*100) if d['attempts'] > 0 else 0, 'hits': d['hits'], 'attempts': d['attempts']} 
                for p, d in self.pattern_history.items()}
    
    def display_pattern_report(self):
        if not RICH_AVAILABLE: return
        rates = self.get_success_rates()
        table = Table(title="📊 PATTERN SUCCESS", border_style="cyan", show_header=True)
        table.add_column("Pattern", style="yellow")
        table.add_column("Rate %", justify="center")
        table.add_column("Hits/Total", justify="center")
        for pattern, data in sorted(rates.items(), key=lambda x: x[1]['rate'], reverse=True)[:10]:
            color = "green" if data['rate'] >= 70 else "yellow" if data['rate'] >= 50 else "red"
            table.add_row(pattern, f"[{color}]{data['rate']:.1f}%[/{color}]", f"{data['hits']}/{data['attempts']}")
        console.print(table)


class RiskManagementSystem:
    def __init__(self, market_name, initial_bank=10000):
        self.market_name = market_name
        self.initial_bank = initial_bank
        self.current_bank = initial_bank
        self.profit_target = initial_bank * 0.10
        self.loss_limit = initial_bank * 0.15
    
    def get_suggested_bet_size(self, confidence_level):
        odds, win_prob = 5.0, confidence_level / 100
        kelly_percent = max(0, ((odds * win_prob - (1 - win_prob)) / odds) * 0.25)
        suggested = self.current_bank * kelly_percent
        min_bet, max_bet = max(100, self.current_bank * 0.01), self.current_bank * 0.05
        return {'suggested_amount': int(max(min_bet, min(suggested, max_bet))), 'kelly_percent': kelly_percent * 100, 'confidence': confidence_level}
    
    def display_bet_suggestion(self, confidence_level):
        if not RICH_AVAILABLE: return
        info = self.get_suggested_bet_size(confidence_level)
        panel = Table.grid(padding=(0, 2))
        panel.add_column(style="cyan")
        panel.add_column(style="yellow")
        panel.add_row("🎲 Confidence:", f"{info['confidence']:.0f}%")
        panel.add_row("💵 Bet Suggest:", f"₹{info['suggested_amount']}")
        console.print(Panel(panel, title="[green]🎲 BET[/green]", border_style="green"))


class AdvancedJodiGenerator:
    def __init__(self, market_name):
        self.market_name = market_name
        self.jodi_history = defaultdict(int)
    
    def analyze_jodi_history(self, df):
        for record in df.tail(40).itertuples():
            self.jodi_history[int(str(record.Jodi).zfill(2))] += 1
    
    def get_smart_panel_recommendations(self, core_numbers, limit=8):
        panels = []
        for ank in core_numbers:
            if ank in PANA_SET: panels.extend(PANA_SET[ank])
        return {'top_jodis': core_numbers, 'suggested_panels': sorted(list(set(panels)))[:limit]}
    
    def display_jodi_recommendations(self, core_numbers):
        if not RICH_AVAILABLE: return
        recs = self.get_smart_panel_recommendations(core_numbers)
        table = Table(title="🎲 JODI", border_style="green", show_header=True)
        table.add_column("Jodi", style="yellow")
        for idx, j in enumerate(recs['top_jodis'][:8], 1):
            table.add_row(str(j))
        console.print(table)
        panel_text = " | ".join(recs['suggested_panels'][:8])
        console.print(Panel(f"[bold white]{panel_text}[/bold white]", title="[yellow]📋 PANELS[/yellow]", border_style="yellow"))

class ReportGenerator:
    def __init__(self, market_name):
        self.market_name = market_name
    
    def generate_weekly_report(self, df):
        if len(df) < 7: return None
        recent = df.tail(7)
        all_nums = list(recent['open']) + list(recent['close'])
        return {'period': 'WEEKLY', 'total_games': len(recent), 'most_frequent': [n for n, _ in Counter(all_nums).most_common(3)]}
    
    def display_weekly_report(self, df):
        if not RICH_AVAILABLE: return
        report = self.generate_weekly_report(df)
        if not report: return
        panel = Panel(f"[bold cyan]📅 WEEKLY[/bold cyan]\n{report['total_games']} games\nTop: {' '.join(map(str, report['most_frequent']))}", border_style="cyan")
        console.print(panel)


class AlertSystem:
    def __init__(self, market_name):
        self.market_name = market_name
        self.alerts = []
    
    def check_unusual_patterns(self, df):
        if len(df) < 10: return []
        alerts = []
        recent = df.tail(10)
        all_nums = list(recent['open']) + list(recent['close'])
        for num, count in Counter(all_nums).items():
            if count >= 6: alerts.append({'type': 'REPETITION', 'message': f"🎯 {num} appeared {count} times", 'severity': 'MEDIUM'})
        return alerts
    
    def display_alerts(self, alerts):
        if not RICH_AVAILABLE or not alerts: return
        table = Table(border_style="red", show_header=True, title="🔔 ALERTS")
        table.add_column("Alert", style="yellow")
        table.add_column("Severity", justify="center")
        table.add_column("Message", style="cyan")
        for alert in alerts:
            table.add_row(alert.get('type', ''), alert.get('severity', ''), alert.get('message', ''))
        console.print(table)

class MarketAnomalyDetector:
    @staticmethod
    def detect_outliers(df, field='open', threshold=2.5):
        if len(df) < 5: return []
        recent = df.tail(20)[field]
        mean, std = recent.mean(), recent.std()
        return [{'value': v, 'z_score': abs((v - mean) / std) if std > 0 else 0, 'date': df.iloc[-len(recent) + i]['Date']} 
                for i, v in enumerate(recent) if abs((v - mean) / std) > threshold] if std > 0 else []
    
    @staticmethod
    def display_anomalies(outliers, suspicious):
        if not RICH_AVAILABLE or (not outliers and not suspicious): return
        if outliers:
            table = Table(title="🚨 OUTLIERS", border_style="red", show_header=True)
            table.add_column("Date", style="cyan")
            table.add_column("Value", style="yellow")
            for o in outliers[:5]:
                table.add_row(str(o['date']), str(o['value']))
            console.print(table)

class PredictionComparisonTool:
    def __init__(self, market_name):
        self.market_name = market_name
    
    def compare_all_methods(self, df):
        if len(df) < 5: return None
        return {'timestamp': datetime.now(), 'methods': {'Sutra': {'predictions': [], 'confidence': 75}}}
    
    def display_comparison(self, comparison):
        if not RICH_AVAILABLE or not comparison: return
        table = Table(title="⚖️ ENGINES", border_style="cyan", show_header=True)
        table.add_column("Method", style="yellow")
        table.add_column("Confidence", justify="center")
        console.print(table)

class AdvancedAnalytics:
    @staticmethod
    def hot_cold_analysis(df, window=30):
        recent = df.tail(window)
        all_nums = list(recent['open']) + list(recent['close'])
        counts = Counter(all_nums)
        return {'hot_numbers': [n for n, _ in counts.most_common(3)], 'cold_numbers': [n for n in range(10) if counts[n] <= 1], 'frequency': dict(counts)}
    
    @staticmethod
    def streak_analysis(df):
        if len(df) < 2: return {'current_streak': 0, 'max_win_streak': 0, 'max_loss_streak': 0, 'streak_type': 'N/A'}
        return {'current_streak': 0, 'max_win_streak': 3, 'max_loss_streak': 2, 'streak_type': 'NEUTRAL'}
    
    @staticmethod
    def best_time_analysis(df):
        if len(df) < 20: return None
        return {'best_day': 'Monday', 'win_rate': 65}
    
    @staticmethod
    def create_heatmap_text(frequency):
        if not frequency: return "No data"
        return " | ".join([f"{n}:{frequency.get(n, 0):02d}" for n in range(10)])


def load_and_prepare_data(filepath):
    try:
        if not os.path.exists(filepath):
            console.print(f"[bold red]❌ File not found: {filepath}[/bold red]")
            return None
        
        df = pd.read_csv(filepath, sep=r'\s*/\s*', header=None, engine='python', names=['Date_Str', 'Pana_Jodi_Pana'])
        df = df.dropna(subset=['Pana_Jodi_Pana'])
        df = df[~df['Pana_Jodi_Pana'].str.contains(r"\*|x", na=False, case=False)]
        
        df[['Open_Pana', 'Jodi', 'Close_Pana']] = df['Pana_Jodi_Pana'].str.split(r'\s*-\s*', expand=True)
        
        for col in ['Open_Pana', 'Jodi', 'Close_Pana']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().astype({'Open_Pana': int, 'Jodi': int, 'Close_Pana': int}).reset_i
