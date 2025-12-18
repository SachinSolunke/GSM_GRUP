import os, sys, json, random, pickle, pandas as pd, numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except:
    RICH_AVAILABLE = False
    class Console:
        def __init__(self): pass
        def print(self, *args, **kwargs): print(*args)
        def input(self, prompt=""): return input(prompt)
    class Panel:
        def __init__(self, content, **kwargs): pass
    class Text:
        @staticmethod
        def from_markup(text): return text
    class Align:
        @staticmethod
        def center(text): return text
    class Rule:
        def __init__(self, **kwargs): pass
    class Table:
        def __init__(self, *args, **kwargs): pass
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass

console = Console()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
ML_MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')

for d in [DATA_DIR, BACKUP_DIR, ML_MODELS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

CUT_ANK = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0, 6: 1, 7: 2, 8: 3, 9: 4}

PANA_SET = {
    1: ['119', '128', '137', '146', '155', '227', '236', '245'], 
    2: ['110', '129', '138', '147', '156', '228', '237', '246'],
    3: ['111', '120', '139', '148', '157', '166', '229', '238'], 
    4: ['112', '130', '149', '158', '167', '220', '239', '248'],
    5: ['113', '122', '140', '159', '168', '177', '230', '249'], 
    6: ['114', '123', '150', '169', '178', '240', '259', '268'],
    7: ['115', '133', '124', '160', '179', '188', '223', '250'], 
    8: ['116', '125', '134', '170', '189', '224', '233', '260'],
    9: ['117', '126', '135', '144', '180', '199', '225', '234'], 
    0: ['118', '127', '136', '190', '226', '235', '244', '299']
}

# ==================== DATA LOADING ====================

def load_data(filepath):
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
        
        df = df.dropna().astype({'Open_Pana': int, 'Jodi': int, 'Close_Pana': int}).reset_index(drop=True)
        df['open'] = df['Jodi'].apply(lambda x: int(str(x).zfill(2)[0]))
        df['close'] = df['Jodi'].apply(lambda x: int(str(x).zfill(2)[1]))
        
        def parse_date(d):
            for fmt in ['%d-%m-%Y', '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y']:
                try:
                    return datetime.strptime(d.strip(), fmt).date()
                except: pass
            return pd.NaT
        
        df['Date'] = df['Date_Str'].apply(parse_date)
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        if len(df) > 0:
            console.print(f"[bold green]✔️ Data loaded: {len(df)} entries[/bold green]")
        return df
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return None

# ==================== CORE ANALYSIS ====================

def get_brahmanda_sutra(df):
    """ब्रह्मांड सूत्र - Get 4 OTC"""
    if len(df) < 1: return None
    
    last = df.iloc[-1]
    close_ank = last['close']
    cut_ank = CUT_ANK[close_ank]
    jodi_sum = (last['open'] + last['close']) % 10
    sum_cut = CUT_ANK[jodi_sum]
    
    otc = sorted(list(set([close_ank, cut_ank, jodi_sum, sum_cut])))
    return otc

def get_jodi_6(df, otc):
    """6 सटीक Jodi - Top 6 Jodis"""
    if len(df) < 10: return []
    
    recent = df.tail(40)
    jodi_counter = Counter(recent['Jodi'])
    top_jodis = [j for j, _ in jodi_counter.most_common(8)]
    
    # Filter by OTC
    filtered = []
    for jodi in top_jodis:
        j_str = str(jodi).zfill(2)
        if int(j_str[0]) in otc or int(j_str[1]) in otc:
            filtered.append(jodi)
    
    return (filtered + top_jodis)[:6]

def get_panne_6(df, otc):
    """6 Panne - Top 6 Panels"""
    if len(df) < 10: return []
    
    recent = df.tail(30)
    opens = list(recent['Open_Pana'])
    closes = list(recent['Close_Pana'])
    
    panels = Counter()
    for p in opens + closes:
        panels[p % 10] += 1
    
    top_panels = [p for p, _ in panels.most_common(6)]
    return top_panels

def get_todays_prediction(df):
    """आज की भविष्यवाणी - Today's Prediction"""
    if len(df) < 5: return {'otc': [], 'jodis': [], 'panels': []}
    
    otc = get_brahmanda_sutra(df)
    if not otc:
        return {'otc': [], 'jodis': [], 'panels': []}
    
    jodis = get_jodi_6(df, otc)
    panels = get_panne_6(df, otc)
    
    return {
        'otc': otc,
        'jodis': jodis,
        'panels': panels
    }

def get_yesterday_result(df):
    """कल का नतीजा - Yesterday's Result"""
    if len(df) < 2: return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    prev_otc = get_brahmanda_sutra(df.iloc[:-1])
    
    return {
        'date': last['Date'],
        'jodi': last['Jodi'],
        'open_pana': last['Open_Pana'],
        'close_pana': last['Close_Pana'],
        'open_ank': last['open'],
        'close_ank': last['close'],
        'predicted_otc': prev_otc if prev_otc else []
    }

def check_prediction_hit(yesterday_result, predicted_otc):
    """चेक करें - Check if prediction was correct"""
    if not yesterday_result or not predicted_otc: return {'ank_hit': False, 'status': 'FAIL'}
    
    open_hit = yesterday_result['open_ank'] in predicted_otc
    close_hit = yesterday_result['close_ank'] in predicted_otc
    
    if open_hit or close_hit:
        hits = []
        if open_hit: hits.append(f"Open:{yesterday_result['open_ank']}")
        if close_hit: hits.append(f"Close:{yesterday_result['close_ank']}")
        return {'ank_hit': True, 'status': '✅ PASS', 'hits': ' / '.join(hits)}
    
    return {'ank_hit': False, 'status': '❌ FAIL'}

# ==================== DISPLAYS ====================

def display_header(market_name):
    """हेडर - Header"""
    if not RICH_AVAILABLE: return
    
    console.print(Panel(
        Align.center(Text.from_markup(
            f'[bold bright_magenta]🔱 {market_name.upper()} 🔱[/bold bright_magenta]\n'
            f'[cyan]PARAMAANU SIMPLE v7.0[/cyan]'
        )),
        border_style="bright_magenta",
        padding=(1, 2)
    ))

def display_yesterday_result(yesterday_result, predicted_otc):
    """कल का नतीजा - Yesterday's Result"""
    if not RICH_AVAILABLE or not yesterday_result: return
    
    try:
        check = check_prediction_hit(yesterday_result, predicted_otc)
        
        table = Table(title="📊 कल का नतीजा (YESTERDAY RESULT)", border_style="cyan", show_header=True)
        table.add_column("Item", style="yellow")
        table.add_column("Value", style="bright_green")
        
        table.add_row("📅 तारीख", str(yesterday_result['date']))
        table.add_row("🎲 जोड़ी", str(yesterday_result['jodi']).zfill(2))
        table.add_row("🔮 Predicted OTC", " ".join(map(str, predicted_otc)))
        table.add_row("📌 अंक Status", check['status'])
        if check['ank_hit']:
            table.add_row("✅ Hits", check['hits'])
        
        console.print(table)
    except: pass

def display_today_prediction(today_pred):
    """आज की भविष्यवाणी - Today's Prediction"""
    if not RICH_AVAILABLE or not today_pred['otc']: return
    
    try:
        # 4 OTC
        console.print(Rule(style="bright_yellow"))
        otc_str = " | ".join(map(str, today_pred['otc']))
        console.print(Panel(
            f"[bold bright_yellow]🎯 आज का संकेत (TODAY'S SIGNAL)[/bold bright_yellow]\n"
            f"[bright_red]4 OTC: {otc_str}[/bright_red]",
            border_style="bright_yellow",
            padding=(1, 2)
        ))
        
        # 6 Jodis
        console.print(Rule(style="green"))
        table = Table(title="🎲 6 सटीक जोड़ियां (6 PRECISE JODIS)", border_style="green", show_header=True)
        table.add_column("No", style="yellow", justify="center")
        table.add_column("Jodi", style="bright_green", justify="center")
        
        for i, jodi in enumerate(today_pred['jodis'][:6], 1):
            table.add_row(str(i), str(jodi).zfill(2))
        
        console.print(table)
        
        # 6 Panels
        console.print(Rule(style="cyan"))
        table = Table(title="📋 6 चुने हुए पन्ने (6 SELECTED PANELS)", border_style="cyan", show_header=True)
        table.add_column("No", style="yellow", justify="center")
        table.add_column("Panel", style="bright_cyan", justify="center")
        
        for i, panel in enumerate(today_pred['panels'][:6], 1):
            table.add_row(str(i), str(panel))
        
        console.print(table)
    except: pass

def display_summary():
    """सारांश - Summary"""
    if not RICH_AVAILABLE: return
    
    console.print(Rule(style="bright_magenta"))
    console.print(Panel(
        "[bold bright_cyan]🙏 धन्यवाद! PARAMAANU SIMPLE v7.0[/bold bright_cyan]\n"
        "[dim]Created with ❤️ by S-T & Aanu-AI[/dim]",
        border_style="bright_magenta",
        padding=(1, 2)
    ))

# ==================== MARKET SELECTION ====================

def select_market():
    """बाजार चुनें - Select Market"""
    if not os.path.exists(DATA_DIR):
        console.print("[bold red]❌ Data directory not found[/bold red]")
        return None
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    market_names = sorted([f.replace('.txt', '') for f in files])
    
    if not market_names:
        console.print("[bold red]❌ No market files found[/bold red]")
        return None
    
    console.print(Rule(style="bold yellow", title="🎯 बाजार चुनें (SELECT MARKET)"))
    
    if RICH_AVAILABLE:
        table = Table(title="📊 उपलब्ध बाजार (AVAILABLE MARKETS)", border_style="magenta", show_header=True)
        table.add_column("No.", style="cyan", justify="center")
        table.add_column("Market", style="green")
        
        for i, name in enumerate(market_names, 1):
            table.add_row(str(i), name.upper())
        
        console.print(table)
    
    console.print("\n[bold cyan]दर्ज करें (Enter number) या [bold]0[/bold] बाहर निकलने के लिए[/bold cyan]")
    
    while True:
        try:
            choice = console.input("\n[bold white]आपकी पसंद (Your choice): [/bold white]").strip()
            
            if choice == '0':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(market_names):
                return market_names[index]
            else:
                console.print("[bold red]❌ गलत नंबर (Invalid number)[/bold red]")
        except:
            console.print("[bold red]❌ दोबारा कोशिश करें (Try again)[/bold red]")

# ==================== MAIN EXECUTION ====================

def run_simple_analysis(market_name):
    """सरल विश्लेषण - Simple Analysis"""
    data_file = os.path.join(DATA_DIR, f"{market_name}.txt")
    
    if RICH_AVAILABLE:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as p:
            t = p.add_task("[cyan]📂 डेटा लोड हो रहा है...[/cyan]", total=None)
            df = load_data(data_file)
            p.update(t, completed=True)
    else:
        df = load_data(data_file)
    
    if df is None or df.empty:
        console.print("[bold red]❌ Data load failed[/bold red]")
        return
    
    if len(df) < 2:
        console.print("[bold red]❌ Insufficient data[/bold red]")
        return
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display Header
    display_header(market_name)
    
    # Get Yesterday's Result
    yesterday_result = get_yesterday_result(df)
    predicted_otc_yesterday = get_brahmanda_sutra(df.iloc[:-1]) if len(df) > 1 else None
    
    # Display Yesterday
    if yesterday_result and predicted_otc_yesterday:
        display_yesterday_result(yesterday_result, predicted_otc_yesterday)
    
    # Get Today's Prediction
    today_pred = get_todays_prediction(df)
    
    # Display Today
    display_today_prediction(today_pred)
    
    # Backup
    backup_data = {
        'market': market_name,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'yesterday': {
            'result': str(yesterday_result) if yesterday_result else None,
            'predicted_otc': predicted_otc_yesterday
        },
        'today': today_pred
    }
    
    try:
        backup_file = os.path.join(BACKUP_DIR, f"{market_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
    except: pass
    
    # Summary
    display_summary()
    
    console.input("\n[bold white]Press ENTER to continue...[/bold white]")

def main():
    """मुख्य - Main"""
    if not RICH_AVAILABLE:
        print("⚠️ Install: pip install rich pandas numpy")
    
    if len(sys.argv) > 1:
        run_simple_analysis(sys.argv[1].strip())
        sys.exit(0)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        market_name = select_market()
        
        if market_name is None:
            console.print("\n[bold cyan]👋 धन्यवाद! Goodbye...[/bold cyan]")
            break
        
        run_simple_analysis(market_name)

if __name__ == "__main__":
    main()
