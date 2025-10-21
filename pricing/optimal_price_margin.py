"""
Optimal Price Determination for Margin Maximization

This script determines the optimal price to maximize margin (profit) based on historical data.

Input data:
- Daily sales volume
- Daily prices
- Daily cost of goods sold (COGS)

The script:
1. Builds a demand elasticity model (how sales depend on price)
2. Optimizes price to maximize total margin (revenue - cost)
3. Visualizes results and provides recommendations

Author: TensorHouse
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class PriceOptimizer:
    """
    A class to optimize pricing based on demand elasticity and cost structure.
    """

    def __init__(self, sales, prices, costs):
        """
        Initialize the optimizer with historical data.

        Parameters:
        -----------
        sales : array-like
            Daily sales volume
        prices : array-like
            Daily prices
        costs : array-like
            Daily cost of goods sold (COGS)
        """
        self.df = pd.DataFrame({
            'sales': np.array(sales),
            'price': np.array(prices),
            'cost': np.array(costs)
        })

        # Calculate derived metrics
        self.df['revenue'] = self.df['sales'] * self.df['price']
        self.df['total_cost'] = self.df['sales'] * self.df['cost']
        self.df['margin'] = self.df['revenue'] - self.df['total_cost']
        self.df['margin_pct'] = (self.df['price'] - self.df['cost']) / self.df['price'] * 100

        self.demand_model = None
        self.optimal_price = None
        self.model_type = None

    def fit_demand_model(self, model_type='linear'):
        """
        Fit a demand model to predict sales based on price.

        Parameters:
        -----------
        model_type : str
            Type of model: 'linear', 'log', 'polynomial'

        Returns:
        --------
        dict : Model performance metrics
        """
        self.model_type = model_type
        X = self.df[['price']].values
        y = self.df['sales'].values

        if model_type == 'linear':
            # Linear demand model: Q = a - b*P
            self.demand_model = LinearRegression()
            self.demand_model.fit(X, y)
            y_pred = self.demand_model.predict(X)

        elif model_type == 'log':
            # Log-log model: log(Q) = a - b*log(P) (constant elasticity)
            X_log = np.log(X + 1e-10)
            y_log = np.log(y + 1e-10)
            self.demand_model = LinearRegression()
            self.demand_model.fit(X_log, y_log)
            y_pred = np.exp(self.demand_model.predict(X_log))

        elif model_type == 'polynomial':
            # Polynomial model: Q = a + b*P + c*P^2
            self.poly_features = PolynomialFeatures(degree=2)
            X_poly = self.poly_features.fit_transform(X)
            self.demand_model = LinearRegression()
            self.demand_model.fit(X_poly, y)
            y_pred = self.demand_model.predict(X_poly)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Calculate performance metrics
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        metrics = {
            'r2_score': r2,
            'mae': mae,
            'model_type': model_type
        }

        print(f"\n{'='*60}")
        print(f"Demand Model: {model_type.upper()}")
        print(f"{'='*60}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f} units")

        if model_type == 'linear':
            slope = self.demand_model.coef_[0]
            intercept = self.demand_model.intercept_
            print(f"Demand equation: Q = {intercept:.2f} + ({slope:.2f}) * P")
            print(f"Price elasticity (at mean): {slope * self.df['price'].mean() / self.df['sales'].mean():.2f}")

        return metrics

    def predict_demand(self, price):
        """
        Predict demand (sales) for a given price.

        Parameters:
        -----------
        price : float or array-like
            Price point(s)

        Returns:
        --------
        float or array : Predicted sales volume
        """
        if self.demand_model is None:
            raise ValueError("Demand model not fitted. Call fit_demand_model() first.")

        price = np.atleast_1d(price)
        X = price.reshape(-1, 1)

        if self.model_type == 'linear':
            demand = self.demand_model.predict(X)
        elif self.model_type == 'log':
            X_log = np.log(X + 1e-10)
            demand = np.exp(self.demand_model.predict(X_log))
        elif self.model_type == 'polynomial':
            X_poly = self.poly_features.transform(X)
            demand = self.demand_model.predict(X_poly)

        # Ensure non-negative demand
        demand = np.maximum(demand, 0)

        return demand if len(demand) > 1 else demand[0]

    def calculate_margin(self, price, cost=None):
        """
        Calculate expected margin for a given price.

        Parameters:
        -----------
        price : float
            Price point
        cost : float, optional
            Cost per unit. If None, uses mean historical cost.

        Returns:
        --------
        float : Expected total margin
        """
        if cost is None:
            cost = self.df['cost'].mean()

        demand = self.predict_demand(price)
        revenue = price * demand
        total_cost = cost * demand
        margin = revenue - total_cost

        return margin

    def optimize_price(self, cost=None, price_range=None, constraints=None):
        """
        Find the optimal price that maximizes margin.

        Parameters:
        -----------
        cost : float, optional
            Cost per unit. If None, uses mean historical cost.
        price_range : tuple, optional
            (min_price, max_price). If None, uses historical range.
        constraints : dict, optional
            Additional constraints:
            - 'min_margin_pct': minimum margin percentage
            - 'min_sales': minimum sales volume

        Returns:
        --------
        dict : Optimization results
        """
        if self.demand_model is None:
            raise ValueError("Demand model not fitted. Call fit_demand_model() first.")

        if cost is None:
            cost = self.df['cost'].mean()

        if price_range is None:
            min_price = max(cost * 1.1, self.df['price'].min() * 0.8)
            max_price = self.df['price'].max() * 1.2
        else:
            min_price, max_price = price_range

        # Objective function (negative because we minimize)
        def objective(price):
            margin = self.calculate_margin(price, cost)

            # Apply constraints as penalties
            penalty = 0
            if constraints:
                if 'min_margin_pct' in constraints:
                    margin_pct = (price - cost) / price * 100
                    if margin_pct < constraints['min_margin_pct']:
                        penalty += 1e6 * (constraints['min_margin_pct'] - margin_pct)

                if 'min_sales' in constraints:
                    sales = self.predict_demand(price)
                    if sales < constraints['min_sales']:
                        penalty += 1e6 * (constraints['min_sales'] - sales)

            return -(margin - penalty)

        # Optimize using bounded method
        result = minimize_scalar(objective, bounds=(min_price, max_price), method='bounded')

        self.optimal_price = result.x
        optimal_demand = self.predict_demand(self.optimal_price)
        optimal_margin = self.calculate_margin(self.optimal_price, cost)
        optimal_margin_pct = (self.optimal_price - cost) / self.optimal_price * 100

        # Compare with current average
        current_avg_price = self.df['price'].mean()
        current_avg_demand = self.df['sales'].mean()
        current_avg_margin = self.df['margin'].mean()

        results = {
            'optimal_price': self.optimal_price,
            'optimal_demand': optimal_demand,
            'optimal_margin': optimal_margin,
            'optimal_margin_pct': optimal_margin_pct,
            'current_avg_price': current_avg_price,
            'current_avg_demand': current_avg_demand,
            'current_avg_margin': current_avg_margin,
            'margin_improvement': optimal_margin - current_avg_margin,
            'margin_improvement_pct': (optimal_margin - current_avg_margin) / current_avg_margin * 100
        }

        print(f"\n{'='*60}")
        print(f"PRICE OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"\nCurrent Average:")
        print(f"  Price: ${current_avg_price:.2f}")
        print(f"  Sales: {current_avg_demand:.2f} units")
        print(f"  Margin: ${current_avg_margin:.2f}")
        print(f"\nOptimal Recommendation:")
        print(f"  Price: ${self.optimal_price:.2f}")
        print(f"  Expected Sales: {optimal_demand:.2f} units")
        print(f"  Expected Margin: ${optimal_margin:.2f}")
        print(f"  Margin %: {optimal_margin_pct:.2f}%")
        print(f"\nImprovement:")
        print(f"  Margin Increase: ${results['margin_improvement']:.2f}")
        print(f"  Margin Increase %: {results['margin_improvement_pct']:.2f}%")
        print(f"  Price Change: ${self.optimal_price - current_avg_price:.2f} ({(self.optimal_price/current_avg_price - 1)*100:.2f}%)")
        print(f"{'='*60}\n")

        return results

    def plot_analysis(self, cost=None, price_range=None, figsize=(15, 10)):
        """
        Create comprehensive visualization of the price optimization analysis.

        Parameters:
        -----------
        cost : float, optional
            Cost per unit for calculations
        price_range : tuple, optional
            (min_price, max_price) for visualization
        figsize : tuple
            Figure size
        """
        if self.demand_model is None:
            raise ValueError("Demand model not fitted. Call fit_demand_model() first.")

        if cost is None:
            cost = self.df['cost'].mean()

        if price_range is None:
            min_price = max(cost * 1.1, self.df['price'].min() * 0.8)
            max_price = self.df['price'].max() * 1.2
        else:
            min_price, max_price = price_range

        # Generate price range for plotting
        prices_plot = np.linspace(min_price, max_price, 100)
        demands_plot = [self.predict_demand(p) for p in prices_plot]
        revenues_plot = prices_plot * demands_plot
        margins_plot = [self.calculate_margin(p, cost) for p in prices_plot]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Price Optimization Analysis', fontsize=16, fontweight='bold')

        # 1. Historical data scatter
        ax1 = axes[0, 0]
        ax1.scatter(self.df['price'], self.df['sales'], alpha=0.6, s=100, label='Historical Data')
        ax1.plot(prices_plot, demands_plot, 'r-', linewidth=2, label='Demand Model')
        ax1.axvline(self.optimal_price, color='g', linestyle='--', linewidth=2, label=f'Optimal Price: ${self.optimal_price:.2f}')
        ax1.set_xlabel('Price ($)', fontsize=12)
        ax1.set_ylabel('Sales Volume', fontsize=12)
        ax1.set_title('Demand Curve', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Margin vs Price
        ax2 = axes[0, 1]
        ax2.plot(prices_plot, margins_plot, 'b-', linewidth=2, label='Total Margin')
        ax2.scatter(self.df['price'], self.df['margin'], alpha=0.4, s=50, color='gray', label='Historical Margins')
        ax2.axvline(self.optimal_price, color='g', linestyle='--', linewidth=2, label=f'Optimal Price: ${self.optimal_price:.2f}')
        optimal_margin = self.calculate_margin(self.optimal_price, cost)
        ax2.axhline(optimal_margin, color='g', linestyle=':', alpha=0.5)
        ax2.scatter([self.optimal_price], [optimal_margin], color='g', s=200, zorder=5, marker='*', label=f'Max Margin: ${optimal_margin:.2f}')
        ax2.set_xlabel('Price ($)', fontsize=12)
        ax2.set_ylabel('Total Margin ($)', fontsize=12)
        ax2.set_title('Margin Optimization Curve', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Revenue vs Price
        ax3 = axes[1, 0]
        ax3.plot(prices_plot, revenues_plot, 'purple', linewidth=2, label='Revenue')
        costs_plot = [self.predict_demand(p) * cost for p in prices_plot]
        ax3.plot(prices_plot, costs_plot, 'orange', linewidth=2, label='Total Cost')
        ax3.fill_between(prices_plot, costs_plot, revenues_plot, where=np.array(revenues_plot) >= np.array(costs_plot), alpha=0.3, color='green', label='Margin Area')
        ax3.axvline(self.optimal_price, color='g', linestyle='--', linewidth=2, label=f'Optimal Price: ${self.optimal_price:.2f}')
        ax3.set_xlabel('Price ($)', fontsize=12)
        ax3.set_ylabel('Amount ($)', fontsize=12)
        ax3.set_title('Revenue vs Cost', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        current_avg_price = self.df['price'].mean()
        current_avg_margin = self.df['margin'].mean()
        optimal_margin = self.calculate_margin(self.optimal_price, cost)
        optimal_demand = self.predict_demand(self.optimal_price)

        summary_text = f"""
        OPTIMIZATION SUMMARY
        {'='*35}

        Current Performance (Average):
          • Price: ${current_avg_price:.2f}
          • Sales: {self.df['sales'].mean():.1f} units/day
          • Margin: ${current_avg_margin:.2f}/day
          • Margin %: {self.df['margin_pct'].mean():.1f}%

        Optimal Recommendation:
          • Price: ${self.optimal_price:.2f}
          • Expected Sales: {optimal_demand:.1f} units/day
          • Expected Margin: ${optimal_margin:.2f}/day
          • Margin %: {(self.optimal_price - cost) / self.optimal_price * 100:.1f}%

        Expected Improvement:
          • Margin increase: ${optimal_margin - current_avg_margin:.2f}/day
          • Margin increase %: {(optimal_margin - current_avg_margin) / current_avg_margin * 100:.1f}%
          • Price change: {(self.optimal_price - current_avg_price) / current_avg_price * 100:+.1f}%

        Model Performance:
          • Type: {self.model_type}
          • Data points: {len(self.df)}
        """

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        return fig

    def sensitivity_analysis(self, cost_range=None, n_points=20):
        """
        Perform sensitivity analysis showing how optimal price changes with cost.

        Parameters:
        -----------
        cost_range : tuple, optional
            (min_cost, max_cost) for analysis
        n_points : int
            Number of points to evaluate

        Returns:
        --------
        DataFrame : Sensitivity analysis results
        """
        if cost_range is None:
            avg_cost = self.df['cost'].mean()
            min_cost = avg_cost * 0.7
            max_cost = avg_cost * 1.3
        else:
            min_cost, max_cost = cost_range

        costs = np.linspace(min_cost, max_cost, n_points)
        results = []

        for cost in costs:
            opt_result = self.optimize_price(cost=cost, price_range=(cost * 1.1, self.df['price'].max() * 1.2))
            results.append({
                'cost': cost,
                'optimal_price': opt_result['optimal_price'],
                'optimal_margin': opt_result['optimal_margin'],
                'margin_pct': opt_result['optimal_margin_pct']
            })

        sensitivity_df = pd.DataFrame(results)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Sensitivity Analysis: Impact of Cost on Optimal Price', fontsize=14, fontweight='bold')

        ax1 = axes[0]
        ax1.plot(sensitivity_df['cost'], sensitivity_df['optimal_price'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Cost per Unit ($)', fontsize=12)
        ax1.set_ylabel('Optimal Price ($)', fontsize=12)
        ax1.set_title('Optimal Price vs Cost', fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(sensitivity_df['cost'], sensitivity_df['optimal_margin'], 'g-', linewidth=2, marker='o')
        ax2.set_xlabel('Cost per Unit ($)', fontsize=12)
        ax2.set_ylabel('Optimal Total Margin ($)', fontsize=12)
        ax2.set_title('Optimal Margin vs Cost', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        return sensitivity_df


def load_data_from_csv(filepath):
    """
    Load data from CSV file.

    Expected columns: 'date', 'sales', 'price', 'cost'

    Parameters:
    -----------
    filepath : str
        Path to CSV file

    Returns:
    --------
    tuple : (sales, prices, costs)
    """
    df = pd.read_csv(filepath)
    return df['sales'].values, df['price'].values, df['cost'].values


def generate_sample_data(n_days=90, base_price=100, base_cost=60):
    """
    Generate sample data for demonstration.

    Parameters:
    -----------
    n_days : int
        Number of days of data
    base_price : float
        Base price level
    base_cost : float
        Base cost level

    Returns:
    --------
    tuple : (sales, prices, costs)
    """
    np.random.seed(42)

    # Generate prices with some variation
    prices = base_price + np.random.normal(0, 10, n_days)
    prices = np.maximum(prices, base_cost * 1.2)  # Ensure price > cost

    # Generate costs with some variation
    costs = base_cost + np.random.normal(0, 3, n_days)
    costs = np.maximum(costs, base_cost * 0.8)

    # Generate sales with negative price elasticity
    # Demand function: Q = 500 - 3*P + noise
    base_demand = 500
    price_sensitivity = -3
    sales = base_demand + price_sensitivity * prices + np.random.normal(0, 20, n_days)
    sales = np.maximum(sales, 0)  # Ensure non-negative

    return sales, prices, costs


if __name__ == "__main__":
    """
    Example usage of the price optimization script.
    """

    print("="*60)
    print("PRICE OPTIMIZATION FOR MARGIN MAXIMIZATION")
    print("="*60)

    # Generate sample data
    print("\n1. Generating sample data...")
    sales, prices, costs = generate_sample_data(n_days=90, base_price=100, base_cost=60)

    print(f"   Data generated: {len(sales)} days of historical data")
    print(f"   Price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}")
    print(f"   Sales range: {np.min(sales):.0f} - {np.max(sales):.0f} units")
    print(f"   Cost range: ${np.min(costs):.2f} - ${np.max(costs):.2f}")

    # Initialize optimizer
    print("\n2. Initializing price optimizer...")
    optimizer = PriceOptimizer(sales, prices, costs)

    # Fit demand model
    print("\n3. Fitting demand model...")
    metrics = optimizer.fit_demand_model(model_type='linear')

    # Optimize price
    print("\n4. Optimizing price for maximum margin...")
    results = optimizer.optimize_price()

    # Visualization
    print("\n5. Generating visualizations...")
    optimizer.plot_analysis()
    plt.savefig('/home/user/tensor-house/pricing/price_optimization_results.png', dpi=150, bbox_inches='tight')
    print("   Saved: price_optimization_results.png")

    # Sensitivity analysis
    print("\n6. Performing sensitivity analysis...")
    sensitivity_df = optimizer.sensitivity_analysis()
    print("\nSensitivity Analysis Sample:")
    print(sensitivity_df.head(10).to_string(index=False))
    plt.savefig('/home/user/tensor-house/pricing/price_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: price_sensitivity_analysis.png")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nUsage with your own data:")
    print("  sales, prices, costs = load_data_from_csv('your_data.csv')")
    print("  optimizer = PriceOptimizer(sales, prices, costs)")
    print("  optimizer.fit_demand_model(model_type='linear')")
    print("  results = optimizer.optimize_price()")
    print("  optimizer.plot_analysis()")
