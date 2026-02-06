"""
Stock Price Prediction using Small Language Model (SLM)
CPU-Optimized Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import math
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Data settings
    STOCK_SYMBOL = "AAPL"  # Stock ticker symbol
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Model settings
    SEQUENCE_LENGTH = 60  # Number of past days to consider
    PREDICTION_DAYS = 1   # Days to predict ahead
    
    # SLM Architecture settings
    D_MODEL = 64          # Model dimension (small for CPU)
    N_HEADS = 4           # Number of attention heads
    N_LAYERS = 2          # Number of transformer layers
    D_FF = 128            # Feed-forward dimension
    DROPOUT = 0.1
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    # Device
    DEVICE = torch.device('cpu')

config = Config()
print(f"🖥️  Running on: {config.DEVICE}")

# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================

class StockDataLoader:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"📊 Fetching data for {self.symbol}...")
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)
        print(f"✅ Downloaded {len(df)} records")
        return df
    
    def prepare_features(self, df):
        """Prepare features for the model"""
        data = pd.DataFrame()
        
        # Price features
        data['Close'] = df['Close']
        data['Open'] = df['Open']
        data['High'] = df['High']
        data['Low'] = df['Low']
        data['Volume'] = df['Volume']
        
        # Technical indicators
        data['SMA_10'] = df['Close'].rolling(window=10).mean()
        data['SMA_20'] = df['Close'].rolling(window=20).mean()
        data['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # Price changes
        data['Price_Change'] = df['Close'].pct_change()
        data['Volatility'] = df['Close'].rolling(window=10).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def create_sequences(self, data, seq_length, pred_days):
        """Create sequences for time series prediction"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(seq_length, len(scaled_data) - pred_days + 1):
            X.append(scaled_data[i-seq_length:i])
            # Predict the Close price (first column)
            y.append(scaled_data[i:i+pred_days, 0])
        
        return np.array(X), np.array(y)

# ============================================
# DATASET CLASS
# ============================================

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# SMALL LANGUAGE MODEL (SLM) ARCHITECTURE
# ============================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class StockSLM(nn.Module):
    """
    Small Language Model for Stock Prediction
    Uses Transformer architecture optimized for CPU
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, 
                 seq_length, pred_days, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length
        self.pred_days = pred_days
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model * seq_length, d_model)
        self.fc2 = nn.Linear(d_model, d_model // 2)
        self.fc3 = nn.Linear(d_model // 2, pred_days)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch, seq_length, input_dim)
        
        # Embed input
        x = self.input_embedding(x)  # (batch, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_length, d_model)
        x = self.layer_norm(x)
        
        # Flatten and predict
        x = x.reshape(x.size(0), -1)  # (batch, seq_length * d_model)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)  # (batch, pred_days)
        
        return x

# ============================================
# ALTERNATIVE: LSTM-BASED SLM (Even lighter)
# ============================================

class StockLSTM_SLM(nn.Module):
    """
    Lightweight LSTM-based model for CPU
    Alternative to Transformer if needed
    """
    def __init__(self, input_dim, hidden_dim, n_layers, pred_days, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_days)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)
        
        output = self.fc(context)  # (batch, pred_days)
        return output

# ============================================
# TRAINING AND EVALUATION
# ============================================

class StockPredictor:
    def __init__(self, config, model_type='transformer'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.data_loader = None
        self.scaler = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def prepare_data(self):
        """Prepare data for training and testing"""
        self.data_loader = StockDataLoader(
            self.config.STOCK_SYMBOL,
            self.config.START_DATE,
            self.config.END_DATE
        )
        
        # Fetch and prepare data
        raw_data = self.data_loader.fetch_data()
        features = self.data_loader.prepare_features(raw_data)
        
        print(f"📈 Features shape: {features.shape}")
        print(f"📈 Features: {list(features.columns)}")
        
        # Create sequences
        X, y = self.data_loader.create_sequences(
            features.values,
            self.config.SEQUENCE_LENGTH,
            self.config.PREDICTION_DAYS
        )
        
        self.scaler = self.data_loader.scaler
        self.n_features = X.shape[2]
        
        # Split data
        split_idx = int(len(X) * self.config.TRAIN_SPLIT)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"🔄 Training samples: {len(X_train)}")
        print(f"🔄 Testing samples: {len(X_test)}")
        
        # Create data loaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )
        
        self.X_test = X_test
        self.y_test = y_test
        self.dates = features.index[self.config.SEQUENCE_LENGTH + split_idx:]
        
        return features
    
    def build_model(self):
        """Build the SLM model"""
        if self.model_type == 'transformer':
            self.model = StockSLM(
                input_dim=self.n_features,
                d_model=self.config.D_MODEL,
                n_heads=self.config.N_HEADS,
                n_layers=self.config.N_LAYERS,
                d_ff=self.config.D_FF,
                seq_length=self.config.SEQUENCE_LENGTH,
                pred_days=self.config.PREDICTION_DAYS,
                dropout=self.config.DROPOUT
            ).to(self.config.DEVICE)
        else:
            self.model = StockLSTM_SLM(
                input_dim=self.n_features,
                hidden_dim=self.config.D_MODEL,
                n_layers=self.config.N_LAYERS,
                pred_days=self.config.PREDICTION_DAYS,
                dropout=self.config.DROPOUT
            ).to(self.config.DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n🤖 Model: {self.model_type.upper()} SLM")
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train(self):
        """Train the model"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print("\n🚀 Starting Training...")
        print("=" * 60)
        
        for epoch in range(self.config.EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.config.DEVICE)
                batch_y = batch_y.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in self.test_loader:
                    batch_X = batch_X.to(self.config.DEVICE)
                    batch_y = batch_y.to(self.config.DEVICE)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(self.test_loader)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.EPOCHS}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("\n✅ Training Complete!")
        print(f"📉 Best Validation Loss: {best_val_loss:.6f}")
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.config.DEVICE)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Inverse transform to get actual prices
        # Create dummy arrays for inverse transform
        n_features = self.scaler.n_features_in_
        
        pred_full = np.zeros((len(predictions), n_features))
        pred_full[:, 0] = predictions.flatten()
        pred_prices = self.scaler.inverse_transform(pred_full)[:, 0]
        
        actual_full = np.zeros((len(actuals), n_features))
        actual_full[:, 0] = actuals.flatten()
        actual_prices = self.scaler.inverse_transform(actual_full)[:, 0]
        
        # Calculate metrics
        mse = mean_squared_error(actual_prices, pred_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, pred_prices)
        r2 = r2_score(actual_prices, pred_prices)
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION METRICS")
        print("=" * 60)
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print("=" * 60)
        
        return pred_prices, actual_prices
    
    def predict_future(self, days=5):
        """Predict future stock prices"""
        self.model.eval()
        
        # Get the last sequence
        last_sequence = self.X_test[-1:].copy()
        last_sequence = torch.FloatTensor(last_sequence).to(self.config.DEVICE)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(days):
                pred = self.model(last_sequence)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                # Update sequence (simplified - just shift and add prediction)
                new_row = last_sequence[0, -1, :].cpu().numpy()
                new_row[0] = pred.cpu().numpy()[0, 0]  # Update close price
                new_row = torch.FloatTensor(new_row).unsqueeze(0).unsqueeze(0)
                
                # Shift sequence
                last_sequence = torch.cat([
                    last_sequence[:, 1:, :],
                    new_row.to(self.config.DEVICE)
                ], dim=1)
        
        # Inverse transform predictions
        n_features = self.scaler.n_features_in_
        pred_full = np.zeros((len(predictions), n_features))
        pred_full[:, 0] = predictions
        future_prices = self.scaler.inverse_transform(pred_full)[:, 0]
        
        return future_prices
    
    def plot_results(self, pred_prices, actual_prices):
        """Plot the results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Predictions vs Actuals
        ax1 = axes[0, 0]
        ax1.plot(actual_prices, label='Actual', color='blue', alpha=0.7)
        ax1.plot(pred_prices, label='Predicted', color='red', alpha=0.7)
        ax1.set_title(f'{self.config.STOCK_SYMBOL} Stock Price Prediction')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training History
        ax2 = axes[0, 1]
        ax2.plot(self.history['train_loss'], label='Train Loss', color='blue')
        ax2.plot(self.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Training History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot
        ax3 = axes[1, 0]
        ax3.scatter(actual_prices, pred_prices, alpha=0.5, color='green')
        ax3.plot([min(actual_prices), max(actual_prices)], 
                 [min(actual_prices), max(actual_prices)], 
                 'r--', label='Perfect Prediction')
        ax3.set_title('Actual vs Predicted')
        ax3.set_xlabel('Actual Price ($)')
        ax3.set_ylabel('Predicted Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Error
        ax4 = axes[1, 1]
        errors = pred_prices - actual_prices
        ax4.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--')
        ax4.set_title('Prediction Error Distribution')
        ax4.set_xlabel('Error ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stock_prediction_results.png', dpi=150)
        plt.show()
        
        print("📊 Results saved to 'stock_prediction_results.png'")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    print("=" * 60)
    print("🚀 STOCK PRICE PREDICTION USING SLM (CPU VERSION)")
    print("=" * 60)
    
    # Initialize predictor
    # Use 'transformer' for Transformer-based SLM
    # Use 'lstm' for lighter LSTM-based model
    predictor = StockPredictor(config, model_type='transformer')
    
    # Prepare data
    features = predictor.prepare_data()
    
    # Build model
    model = predictor.build_model()
    print(f"\n📐 Model Architecture:\n{model}")
    
    # Train model
    predictor.train()
    
    # Evaluate
    pred_prices, actual_prices = predictor.evaluate()
    
    # Future predictions
    future_days = 5
    future_prices = predictor.predict_future(days=future_days)
    
    print(f"\n🔮 FUTURE PRICE PREDICTIONS ({future_days} days):")
    print("-" * 40)
    last_actual = actual_prices[-1]
    print(f"Last Known Price: ${last_actual:.2f}")
    for i, price in enumerate(future_prices, 1):
        change = ((price - last_actual) / last_actual) * 100
        arrow = "📈" if change > 0 else "📉"
        print(f"Day {i}: ${price:.2f} ({arrow} {change:+.2f}%)")
    
    # Plot results
    predictor.plot_results(pred_prices, actual_prices)
    
    return predictor

if __name__ == "__main__":
    predictor = main()
