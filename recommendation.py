import streamlit as st
import pandas as pd


# --------------------------------------------
# Customer Search
# --------------------------------------------

def customer_search(df):

    st.header("🔍 Customer Search")

    customer_ids = df["Customer_ID"].astype(str).tolist()

    selected = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    if selected:

        customer = df[df["Customer_ID"].astype(str)==selected]

        st.dataframe(customer)

        st.success(
            f"Customer Segment : {customer.iloc[0]['Customer_Type']}"
        )


# --------------------------------------------
# Business Recommendation
# --------------------------------------------

def business_recommendation(df):

    st.header("💡 Business Insights")

    segment = st.selectbox(
        "Choose Segment",
        ["High Value","Medium Value","Low Value"]
    )

    if segment=="High Value":

        st.success("""
### Recommended Strategy

✔ VIP Membership

✔ Premium Services

✔ Loyalty Rewards

✔ Early Product Access

✔ Personalized Offers

✔ Referral Bonus

Expected Outcome:

Increase Lifetime Value
""")

    elif segment=="Medium Value":

        st.info("""
### Recommended Strategy

✔ Bundle Offers

✔ Cross Selling

✔ Email Campaign

✔ Festival Discounts

✔ Personalized Recommendations

Expected Outcome:

Convert Medium Customers to High Value
""")

    else:

        st.warning("""
### Recommended Strategy

✔ Coupon Campaign

✔ Cashback

✔ Welcome Offers

✔ SMS Reminder

✔ First Purchase Discount

Expected Outcome:

Increase Purchase Frequency
""")