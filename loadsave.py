"""
Stock Price Prediction using SLM
- Train new model
- Save trained model
- Load and predict with saved model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    STOCK_SYMBOL = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    SEQUENCE_LENGTH = 60
    PREDICTION_DAYS = 1
    
    # Model settings (small for CPU)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    DEVICE = torch.device('cpu')
    MODEL_PATH = "stock_model.pth"
    SCALER_PATH = "scaler.pkl"

config = Config()

# ============================================
# MODEL ARCHITECTURE
# ============================================

class StockSLM(nn.Module):
    """Small Language Model for Stock Prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(StockSLM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        out = self.fc(context)
        return out

# ============================================
# DATA PROCESSOR
# ============================================

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.features = None
        
    def download_data(self):
        """Download stock data from Yahoo Finance"""
        print(f"📥 Downloading {self.config.STOCK_SYMBOL} data...")
        self.data = yf.download(
            self.config.STOCK_SYMBOL,
            start=self.config.START_DATE,
            end=self.config.END_DATE,
            progress=False
        )
        print(f"✅ Downloaded {len(self.data)} days of data")
        print(f"📅 Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        return self.data
    
    def create_features(self):
        """Create technical indicators as features"""
        df = self.data.copy()
        
        features = pd.DataFrame()
        features['Close'] = df['Close']
        features['Open'] = df['Open']
        features['High'] = df['High']
        features['Low'] = df['Low']
        features['Volume'] = df['Volume']
        
        # Moving averages
        features['SMA_10'] = df['Close'].rolling(window=10).mean()
        features['SMA_20'] = df['Close'].rolling(window=20).mean()
        features['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # Price changes
        features['Daily_Return'] = df['Close'].pct_change()
        features['Volatility'] = df['Close'].rolling(window=10).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        features['MACD'] = exp1 - exp2
        
        # Drop NaN
        features = features.dropna()
        self.features = features
        
        print(f"📊 Created {len(features.columns)} features")
        return features
    
    def prepare_sequences(self):
        """Create sequences for training"""
        data = self.features.values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.config.SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i-self.config.SEQUENCE_LENGTH:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        X = np.array(X)
        y = np.array(y)
        
        # Split
        split = int(len(X) * self.config.TRAIN_SPLIT)
        
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📉 Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, self.features.index[self.config.SEQUENCE_LENGTH + split:]
    
    def save_scaler(self):
        """Save scaler for later use"""
        with open(self.config.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Scaler saved to {self.config.SCALER_PATH}")
    
    def load_scaler(self):
        """Load saved scaler"""
        with open(self.config.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"📂 Scaler loaded from {self.config.SCALER_PATH}")

# ============================================
# TRAINER
# ============================================

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train(self, X_train, y_train, X_test, y_test):
        """Train the model"""
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test)
        
        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_loss = float('inf')
        
        print("\n" + "="*50)
        print("🚀 TRAINING STARTED")
        print("="*50)
        
        for epoch in range(self.config.EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_test_t)
                val_loss = criterion(val_output.squeeze(), y_test_t).item()
            
            scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.EPOCHS} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
        print("\n✅ Training complete!")
        print(f"💾 Best model saved to {self.config.MODEL_PATH}")
        
        return self.history

# ============================================
# PREDICTOR
# ============================================

class Predictor:
    def __init__(self, model, scaler, config):
        self.model = model
        self.scaler = scaler
        self.config = config
    
    def predict(self, X_test):
        """Make predictions on test data"""
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            predictions = self.model(X_test_t).squeeze().numpy()
        
        return predictions
    
    def inverse_transform(self, scaled_values):
        """Convert scaled values back to actual prices"""
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(scaled_values), n_features))
        dummy[:, 0] = scaled_values
        return self.scaler.inverse_transform(dummy)[:, 0]
    
    def predict_next_day(self, last_sequence):
        """Predict next day's price"""
        self.model.eval()
        
        # Scale the sequence
        scaled_seq = self.scaler.transform(last_sequence)
        X = torch.FloatTensor(scaled_seq).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = self.model(X).item()
        
        # Inverse transform
        pred_price = self.inverse_transform([pred_scaled])[0]
        
        return pred_price
    
    def predict_multiple_days(self, last_sequence, days=5):
        """Predict multiple days ahead"""
        predictions = []
        current_seq = last_sequence.copy()
        
        for _ in range(days):
            # Predict next day
            next_pred = self.predict_next_day(current_seq)
            predictions.append(next_pred)
            
            # Update sequence (simplified approach)
            new_row = current_seq[-1].copy()
            new_row[0] = next_pred  # Update close price
            current_seq = np.vstack([current_seq[1:], new_row])
        
        return predictions

# ============================================
# MAIN CLASS - EASY TO USE
# ============================================

class StockPredictionSystem:
    """
    Main class for stock prediction
    
    Usage:
        system = StockPredictionSystem("AAPL")
        
        # Option 1: Train new model
        system.train_new_model()
        
        # Option 2: Load existing model
        system.load_model()
        
        # Make predictions
        system.predict_future(days=5)
    """
    
    def __init__(self, stock_symbol="AAPL"):
        self.config = Config()
        self.config.STOCK_SYMBOL = stock_symbol
        self.config.MODEL_PATH = f"{stock_symbol}_model.pth"
        self.config.SCALER_PATH = f"{stock_symbol}_scaler.pkl"
        
        self.processor = DataProcessor(self.config)
        self.model = None
        self.predictor = None
        self.is_trained = False
        
        print(f"📊 Stock Prediction System for {stock_symbol}")
        print(f"🖥️  Running on: {self.config.DEVICE}")
    
    def train_new_model(self):
        """Train a new model from scratch"""
        print("\n" + "="*50)
        print("📚 TRAINING NEW MODEL")
        print("="*50)
        
        # Step 1: Download and process data
        self.processor.download_data()
        self.processor.create_features()
        X_train, X_test, y_train, y_test, self.test_dates = self.processor.prepare_sequences()
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Step 2: Create model
        n_features = X_train.shape[2]
        self.model = StockSLM(
            input_size=n_features,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            output_size=1,
            dropout=self.config.DROPOUT
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🤖 Model created with {total_params:,} parameters")
        
        # Step 3: Train
        trainer = Trainer(self.model, self.config)
        self.history = trainer.train(X_train, y_train, X_test, y_test)
        
        # Step 4: Save scaler
        self.processor.save_scaler()
        
        # Step 5: Create predictor
        self.predictor = Predictor(self.model, self.processor.scaler, self.config)
        self.is_trained = True
        
        # Step 6: Evaluate
        self.evaluate()
        
        return self
    
    def load_model(self):
        """Load a previously trained model"""
        print("\n" + "="*50)
        print("📂 LOADING SAVED MODEL")
        print("="*50)
        
        # Check if files exist
        if not os.path.exists(self.config.MODEL_PATH):
            print(f"❌ Model file not found: {self.config.MODEL_PATH}")
            print("Please train a model first using train_new_model()")
            return None
        
        if not os.path.exists(self.config.SCALER_PATH):
            print(f"❌ Scaler file not found: {self.config.SCALER_PATH}")
            return None
        
        # Download latest data for features
        self.processor.download_data()
        self.processor.create_features()
        
        # Load scaler
        self.processor.load_scaler()
        
        # Create and load model
        n_features = len(self.processor.features.columns)
        self.model = StockSLM(
            input_size=n_features,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            output_size=1,
            dropout=self.config.DROPOUT
        )
        
        self.model.load_state_dict(torch.load(self.config.MODEL_PATH))
        self.model.eval()
        
        print(f"✅ Model loaded from {self.config.MODEL_PATH}")
        
        # Create predictor
        self.predictor = Predictor(self.model, self.processor.scaler, self.config)
        self.is_trained = True
        
        return self
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_trained:
            print("❌ Model not trained. Please train or load a model first.")
            return
        
        # Make predictions
        predictions_scaled = self.predictor.predict(self.X_test)
        
        # Inverse transform
        pred_prices = self.predictor.inverse_transform(predictions_scaled)
        actual_prices = self.predictor.inverse_transform(self.y_test)
        
        # Calculate metrics
        mse = mean_squared_error(actual_prices, pred_prices)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_prices, pred_prices)
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION")
        print("="*50)
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE:  ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print("="*50)
        
        self.pred_prices = pred_prices
        self.actual_prices = actual_prices
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}
    
    def predict_future(self, days=5):
        """Predict future stock prices"""
        if not self.is_trained:
            print("❌ Model not trained. Please train or load a model first.")
            return None
        
        # Get last sequence
        last_sequence = self.processor.features.values[-self.config.SEQUENCE_LENGTH:]
        
        # Current price
        current_price = self.processor.features['Close'].iloc[-1]
        current_date = self.processor.features.index[-1]
        
        # Predict
        future_prices = self.predictor.predict_multiple_days(last_sequence, days)
        
        print("\n" + "="*50)
        print(f"🔮 FUTURE PRICE PREDICTIONS - {self.config.STOCK_SYMBOL}")
        print("="*50)
        print(f"Current Price ({current_date.date()}): ${current_price:.2f}")
        print("-"*50)
        
        results = []
        for i, price in enumerate(future_prices, 1):
            future_date = current_date + timedelta(days=i)
            change = ((price - current_price) / current_price) * 100
            arrow = "📈" if change > 0 else "📉"
            
            print(f"Day {i} ({future_date.date()}): ${price:.2f} {arrow} ({change:+.2f}%)")
            results.append({
                'day': i,
                'date': future_date.date(),
                'price': price,
                'change': change
            })
        
        print("="*50)
        
        return results
    
    def plot_results(self):
        """Plot prediction results"""
        if not hasattr(self, 'pred_prices'):
            print("❌ No results to plot. Please run evaluate() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Predictions vs Actual
        ax1 = axes[0, 0]
        ax1.plot(self.actual_prices, label='Actual', color='blue', linewidth=1.5)
        ax1.plot(self.pred_prices, label='Predicted', color='red', linewidth=1.5, alpha=0.7)
        ax1.set_title(f'{self.config.STOCK_SYMBOL} - Actual vs Predicted Prices')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training History
        if hasattr(self, 'history'):
            ax2 = axes[0, 1]
            ax2.plot(self.history['train_loss'], label='Train Loss')
            ax2.plot(self.history['val_loss'], label='Validation Loss')
            ax2.set_title('Training History')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot
        ax3 = axes[1, 0]
        ax3.scatter(self.actual_prices, self.pred_prices, alpha=0.5, color='green')
        min_val = min(min(self.actual_prices), min(self.pred_prices))
        max_val = max(max(self.actual_prices), max(self.pred_prices))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax3.set_title('Actual vs Predicted Scatter')
        ax3.set_xlabel('Actual Price ($)')
        ax3.set_ylabel('Predicted Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution
        ax4 = axes[1, 1]
        errors = self.pred_prices - self.actual_prices
        ax4.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Prediction Error Distribution')
        ax4.set_xlabel('Error ($)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.STOCK_SYMBOL}_results.png', dpi=150)
        plt.show()
        
        print(f"📊 Plot saved to {self.config.STOCK_SYMBOL}_results.png")

# ============================================
# HOW TO USE
# ============================================

def main():
    """
    Main function demonstrating how to use the system
    """
    
    print("="*60)
    print("🚀 STOCK PRICE PREDICTION USING SLM")
    print("="*60)
    
    # Choose your stock
    stock_symbol = "AAPL"  # Change this to any stock symbol
    
    # Create system
    system = StockPredictionSystem(stock_symbol)
    
    # ============================================
    # OPTION 1: TRAIN A NEW MODEL
    # ============================================
    print("\n📝 Training new model...")
    system.train_new_model()
    
    # ============================================
    # OPTION 2: LOAD EXISTING MODEL (uncomment to use)
    # ============================================
    # system.load_model()
    
    # ============================================
    # PREDICT FUTURE PRICES
    # ============================================
    future_predictions = system.predict_future(days=5)
    
    # ============================================
    # PLOT RESULTS
    # ============================================
    system.plot_results()
    
    return system

# ============================================
# RUN THE CODE
# ============================================

if __name__ == "__main__":
    system = main()
