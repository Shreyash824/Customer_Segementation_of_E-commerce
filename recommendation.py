import streamlit as st
import pandas as pd

# --------------------------------------------
# Customer Search
# --------------------------------------------

def customer_search(df):

    st.header("🔍 Customer Search")

    # Find Customer ID column automatically
    id_columns = [c for c in df.columns if "customer" in c.lower()]

    if len(id_columns) == 0:
        st.warning("No Customer ID column found.")
        return

    id_col = id_columns[0]

    customer_ids = df[id_col].astype(str).tolist()

    selected = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    if selected:

        customer = df[df[id_col].astype(str) == selected]

        st.dataframe(customer, use_container_width=True)

        st.success(
            f"Customer Segment: {customer.iloc[0]['Customer_Type']}"
        )


# --------------------------------------------
# Business Recommendation
# --------------------------------------------

def business_recommendation(df):

    st.header("💡 Business Insights")

    segment = st.selectbox(
        "Choose Segment",
        ["High Value", "Medium Value", "Low Value"]
    )

    if segment == "High Value":

        st.success("""
### Recommended Strategy

- ⭐ VIP Membership
- ⭐ Premium Services
- ⭐ Loyalty Rewards
- ⭐ Early Product Access
- ⭐ Personalized Offers
- ⭐ Referral Bonus

**Expected Outcome:** Increase Customer Lifetime Value
""")

    elif segment == "Medium Value":

        st.info("""
### Recommended Strategy

- 📦 Bundle Offers
- 📦 Cross Selling
- 📦 Email Campaign
- 📦 Festival Discounts
- 📦 Personalized Recommendations

**Expected Outcome:** Convert Medium Value Customers into High Value Customers
""")

    else:

        st.warning("""
### Recommended Strategy

- 🎁 Coupon Campaign
- 🎁 Cashback
- 🎁 Welcome Offers
- 🎁 SMS Reminder
- 🎁 First Purchase Discount

**Expected Outcome:** Increase Purchase Frequency
""")
