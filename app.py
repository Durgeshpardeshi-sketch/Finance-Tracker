import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Page Config
st.set_page_config(page_title="AI Expense Tracker", layout="wide")

st.title("💰 Personal AI Expense Manager")
st.markdown("Upload your bank statement and let AI categorize your spending!")

# --- 1. AI Logic Function ---
@st.cache_resource # This makes the app fast
def train_ai():
    train_df = pd.read_csv('transactions.csv')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(train_df['description'])
    y = train_df['category']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_ai()

# --- 2. Sidebar for Upload ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_saver = st.sidebar.file_uploader("Upload 'test_statement.csv'", type=["csv"])

if uploaded_file is not None:
    # Load user data
    user_df = pd.read_csv(uploaded_file)
    
    # Predict Categories
    user_df['predicted_category'] = model.predict(vectorizer.transform(user_df['description']))
    
    # --- 3. Dashboard Metrics ---
    total_spent = user_df['amount'].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Expenses", f"₹{total_spent:,}")
    col2.metric("Transactions", len(user_df))
    col3.metric("Top Category", user_df.groupby('predicted_category')['amount'].sum().idxmax())

    # --- 4. Charts Section ---
    st.subheader("Spending Analysis")
    c1, c2 = st.columns(2)
    
    # Pie Chart
    category_totals = user_df.groupby('predicted_category')['amount'].sum().reset_index()
    fig_pie = px.pie(category_totals, values='amount', names='predicted_category', title="Expenses by Category")
    c1.plotly_chart(fig_pie)
    
    # Bar Chart
    fig_bar = px.bar(category_totals, x='predicted_category', y='amount', color='predicted_category', title="Budget Breakdown")
    c2.plotly_chart(fig_bar)

    # --- 5. Data Table ---
    st.subheader("Categorized Transactions")
    st.dataframe(user_df, use_container_width=True)

else:
    st.info("👈 Please upload the 'test_statement.csv' file from the sidebar to see the magic!")