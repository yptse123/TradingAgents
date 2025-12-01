"""
Asset Universes - Comprehensive lists of investable assets across different categories
"""

# Large Cap US Stocks
STOCK_UNIVERSE = {
    "mega_cap": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    ],
    "large_cap_growth": [
        "CRM", "ADBE", "AMD", "NFLX", "COST", "AVGO", "QCOM", "TXN", "INTC", "PYPL",
        "NOW", "SHOP", "SQ", "SNOW", "UBER", "ABNB", "DDOG", "ZS", "PANW", "CRWD",
    ],
    "large_cap_value": [
        "PEP", "KO", "PFE", "TMO", "ABT", "DHR", "ACN", "CSCO", "VZ", "T",
        "IBM", "GE", "HON", "MMM", "CAT", "DE", "BA", "LMT", "RTX", "GD",
    ],
    "dividend_aristocrats": [
        "JNJ", "PG", "KO", "PEP", "MMM", "ABT", "ABBV", "XOM", "CVX", "CL",
        "ED", "EMR", "GPC", "ITW", "LOW", "MCD", "SWK", "TGT", "WMT", "AFL",
    ],
    "mid_cap_growth": [
        "FTNT", "TEAM", "OKTA", "TTD", "ZM", "ROKU", "PINS", "SNAP", "TWLO", "NET",
        "BILL", "HUBS", "VEEV", "SPLK", "MDB", "ESTC", "FSLY", "PATH", "DOCN", "CFLT",
    ],
}

# ETF Universe - Broad Market
ETF_UNIVERSE = {
    "total_market": [
        {"ticker": "SPY", "name": "SPDR S&P 500", "category": "US Large Cap"},
        {"ticker": "QQQ", "name": "Invesco QQQ Trust", "category": "US Tech/Growth"},
        {"ticker": "IWM", "name": "iShares Russell 2000", "category": "US Small Cap"},
        {"ticker": "VTI", "name": "Vanguard Total Stock Market", "category": "US Total Market"},
        {"ticker": "VOO", "name": "Vanguard S&P 500", "category": "US Large Cap"},
        {"ticker": "IVV", "name": "iShares Core S&P 500", "category": "US Large Cap"},
        {"ticker": "DIA", "name": "SPDR Dow Jones", "category": "US Large Cap Value"},
        {"ticker": "MDY", "name": "SPDR S&P MidCap 400", "category": "US Mid Cap"},
    ],
    "factor_etfs": [
        {"ticker": "VTV", "name": "Vanguard Value", "category": "US Value"},
        {"ticker": "VUG", "name": "Vanguard Growth", "category": "US Growth"},
        {"ticker": "MTUM", "name": "iShares Momentum", "category": "US Momentum"},
        {"ticker": "QUAL", "name": "iShares Quality", "category": "US Quality"},
        {"ticker": "USMV", "name": "iShares Min Vol", "category": "US Low Volatility"},
        {"ticker": "VYM", "name": "Vanguard High Dividend", "category": "US Dividend"},
        {"ticker": "SCHD", "name": "Schwab US Dividend", "category": "US Dividend"},
        {"ticker": "DVY", "name": "iShares Select Dividend", "category": "US Dividend"},
    ],
}

# Bond ETF Universe
BOND_ETF_UNIVERSE = {
    "government_bonds": [
        {"ticker": "TLT", "name": "iShares 20+ Year Treasury", "duration": "Long", "risk": "Low"},
        {"ticker": "IEF", "name": "iShares 7-10 Year Treasury", "duration": "Intermediate", "risk": "Low"},
        {"ticker": "SHY", "name": "iShares 1-3 Year Treasury", "duration": "Short", "risk": "Very Low"},
        {"ticker": "GOVT", "name": "iShares US Treasury", "duration": "Mixed", "risk": "Low"},
        {"ticker": "TIP", "name": "iShares TIPS", "duration": "Mixed", "risk": "Low"},
        {"ticker": "STIP", "name": "iShares 0-5 Year TIPS", "duration": "Short", "risk": "Very Low"},
    ],
    "corporate_bonds": [
        {"ticker": "LQD", "name": "iShares Investment Grade Corporate", "duration": "Intermediate", "risk": "Low-Medium"},
        {"ticker": "VCIT", "name": "Vanguard Intermediate Corporate", "duration": "Intermediate", "risk": "Low-Medium"},
        {"ticker": "VCSH", "name": "Vanguard Short-Term Corporate", "duration": "Short", "risk": "Low"},
        {"ticker": "IGIB", "name": "iShares 5-10 Year IG Corporate", "duration": "Intermediate", "risk": "Low-Medium"},
    ],
    "high_yield_bonds": [
        {"ticker": "HYG", "name": "iShares High Yield Corporate", "duration": "Mixed", "risk": "Medium-High"},
        {"ticker": "JNK", "name": "SPDR High Yield Bond", "duration": "Mixed", "risk": "Medium-High"},
        {"ticker": "USHY", "name": "iShares Broad USD High Yield", "duration": "Mixed", "risk": "Medium-High"},
    ],
    "municipal_bonds": [
        {"ticker": "MUB", "name": "iShares National Muni Bond", "duration": "Intermediate", "risk": "Low"},
        {"ticker": "VTEB", "name": "Vanguard Tax-Exempt Bond", "duration": "Intermediate", "risk": "Low"},
        {"ticker": "HYD", "name": "VanEck High Yield Muni", "duration": "Mixed", "risk": "Medium"},
    ],
    "aggregate_bonds": [
        {"ticker": "AGG", "name": "iShares Core US Aggregate", "duration": "Intermediate", "risk": "Low"},
        {"ticker": "BND", "name": "Vanguard Total Bond Market", "duration": "Intermediate", "risk": "Low"},
        {"ticker": "SCHZ", "name": "Schwab US Aggregate Bond", "duration": "Intermediate", "risk": "Low"},
    ],
}

# Sector ETFs
SECTOR_ETF_UNIVERSE = {
    "technology": [
        {"ticker": "XLK", "name": "Technology Select Sector SPDR", "sector": "Technology"},
        {"ticker": "VGT", "name": "Vanguard Information Technology", "sector": "Technology"},
        {"ticker": "SOXX", "name": "iShares Semiconductor", "sector": "Semiconductors"},
        {"ticker": "IGV", "name": "iShares Software", "sector": "Software"},
        {"ticker": "ARKK", "name": "ARK Innovation", "sector": "Disruptive Tech"},
    ],
    "healthcare": [
        {"ticker": "XLV", "name": "Health Care Select Sector SPDR", "sector": "Healthcare"},
        {"ticker": "VHT", "name": "Vanguard Health Care", "sector": "Healthcare"},
        {"ticker": "IBB", "name": "iShares Biotechnology", "sector": "Biotech"},
        {"ticker": "XBI", "name": "SPDR S&P Biotech", "sector": "Biotech"},
    ],
    "financials": [
        {"ticker": "XLF", "name": "Financial Select Sector SPDR", "sector": "Financials"},
        {"ticker": "VFH", "name": "Vanguard Financials", "sector": "Financials"},
        {"ticker": "KRE", "name": "SPDR S&P Regional Banking", "sector": "Banks"},
        {"ticker": "KBE", "name": "SPDR S&P Bank", "sector": "Banks"},
    ],
    "energy": [
        {"ticker": "XLE", "name": "Energy Select Sector SPDR", "sector": "Energy"},
        {"ticker": "VDE", "name": "Vanguard Energy", "sector": "Energy"},
        {"ticker": "XOP", "name": "SPDR S&P Oil & Gas", "sector": "Oil & Gas"},
        {"ticker": "OIH", "name": "VanEck Oil Services", "sector": "Oil Services"},
    ],
    "consumer": [
        {"ticker": "XLY", "name": "Consumer Discretionary SPDR", "sector": "Consumer Discretionary"},
        {"ticker": "XLP", "name": "Consumer Staples SPDR", "sector": "Consumer Staples"},
        {"ticker": "VCR", "name": "Vanguard Consumer Discretionary", "sector": "Consumer Discretionary"},
        {"ticker": "VDC", "name": "Vanguard Consumer Staples", "sector": "Consumer Staples"},
    ],
    "industrials": [
        {"ticker": "XLI", "name": "Industrial Select Sector SPDR", "sector": "Industrials"},
        {"ticker": "VIS", "name": "Vanguard Industrials", "sector": "Industrials"},
        {"ticker": "ITA", "name": "iShares US Aerospace & Defense", "sector": "Defense"},
    ],
    "utilities": [
        {"ticker": "XLU", "name": "Utilities Select Sector SPDR", "sector": "Utilities"},
        {"ticker": "VPU", "name": "Vanguard Utilities", "sector": "Utilities"},
    ],
    "real_estate": [
        {"ticker": "XLRE", "name": "Real Estate Select Sector SPDR", "sector": "Real Estate"},
        {"ticker": "VNQ", "name": "Vanguard Real Estate", "sector": "REITs"},
        {"ticker": "IYR", "name": "iShares US Real Estate", "sector": "REITs"},
    ],
    "materials": [
        {"ticker": "XLB", "name": "Materials Select Sector SPDR", "sector": "Materials"},
        {"ticker": "VAW", "name": "Vanguard Materials", "sector": "Materials"},
        {"ticker": "GDX", "name": "VanEck Gold Miners", "sector": "Gold Miners"},
    ],
}

# Commodity ETFs
COMMODITY_UNIVERSE = {
    "precious_metals": [
        {"ticker": "GLD", "name": "SPDR Gold Shares", "commodity": "Gold"},
        {"ticker": "IAU", "name": "iShares Gold Trust", "commodity": "Gold"},
        {"ticker": "SLV", "name": "iShares Silver Trust", "commodity": "Silver"},
        {"ticker": "PPLT", "name": "abrdn Platinum", "commodity": "Platinum"},
    ],
    "energy_commodities": [
        {"ticker": "USO", "name": "United States Oil Fund", "commodity": "Crude Oil"},
        {"ticker": "UNG", "name": "United States Natural Gas", "commodity": "Natural Gas"},
        {"ticker": "DBE", "name": "Invesco DB Energy", "commodity": "Energy Basket"},
    ],
    "broad_commodities": [
        {"ticker": "DBC", "name": "Invesco DB Commodity Index", "commodity": "Broad Basket"},
        {"ticker": "GSG", "name": "iShares S&P GSCI", "commodity": "Broad Basket"},
        {"ticker": "PDBC", "name": "Invesco Optimum Yield Diversified", "commodity": "Broad Basket"},
    ],
    "agriculture": [
        {"ticker": "DBA", "name": "Invesco DB Agriculture", "commodity": "Agriculture"},
        {"ticker": "CORN", "name": "Teucrium Corn Fund", "commodity": "Corn"},
        {"ticker": "WEAT", "name": "Teucrium Wheat Fund", "commodity": "Wheat"},
    ],
}

# REITs
REIT_UNIVERSE = {
    "diversified": [
        {"ticker": "VNQ", "name": "Vanguard Real Estate", "focus": "Diversified"},
        {"ticker": "IYR", "name": "iShares US Real Estate", "focus": "Diversified"},
        {"ticker": "SCHH", "name": "Schwab US REIT", "focus": "Diversified"},
    ],
    "residential": [
        {"ticker": "REZ", "name": "iShares Residential & Multisector", "focus": "Residential"},
        {"ticker": "EQR", "name": "Equity Residential", "focus": "Apartments"},
        {"ticker": "AVB", "name": "AvalonBay Communities", "focus": "Apartments"},
    ],
    "commercial": [
        {"ticker": "SPG", "name": "Simon Property Group", "focus": "Malls"},
        {"ticker": "PLD", "name": "Prologis", "focus": "Industrial"},
        {"ticker": "AMT", "name": "American Tower", "focus": "Cell Towers"},
        {"ticker": "CCI", "name": "Crown Castle", "focus": "Cell Towers"},
        {"ticker": "EQIX", "name": "Equinix", "focus": "Data Centers"},
        {"ticker": "DLR", "name": "Digital Realty", "focus": "Data Centers"},
    ],
    "healthcare_reits": [
        {"ticker": "WELL", "name": "Welltower", "focus": "Healthcare"},
        {"ticker": "VTR", "name": "Ventas", "focus": "Healthcare"},
        {"ticker": "HCP", "name": "Healthpeak Properties", "focus": "Healthcare"},
    ],
}

# International ETFs
INTERNATIONAL_UNIVERSE = {
    "developed_markets": [
        {"ticker": "EFA", "name": "iShares EAFE", "region": "Developed ex-US"},
        {"ticker": "VEA", "name": "Vanguard FTSE Developed", "region": "Developed ex-US"},
        {"ticker": "IEFA", "name": "iShares Core MSCI EAFE", "region": "Developed ex-US"},
        {"ticker": "EWJ", "name": "iShares MSCI Japan", "region": "Japan"},
        {"ticker": "EWG", "name": "iShares MSCI Germany", "region": "Germany"},
        {"ticker": "EWU", "name": "iShares MSCI UK", "region": "UK"},
        {"ticker": "EWC", "name": "iShares MSCI Canada", "region": "Canada"},
        {"ticker": "EWA", "name": "iShares MSCI Australia", "region": "Australia"},
    ],
    "emerging_markets": [
        {"ticker": "EEM", "name": "iShares MSCI Emerging Markets", "region": "Emerging Markets"},
        {"ticker": "VWO", "name": "Vanguard FTSE Emerging Markets", "region": "Emerging Markets"},
        {"ticker": "IEMG", "name": "iShares Core MSCI EM", "region": "Emerging Markets"},
        {"ticker": "FXI", "name": "iShares China Large-Cap", "region": "China"},
        {"ticker": "EWZ", "name": "iShares MSCI Brazil", "region": "Brazil"},
        {"ticker": "INDA", "name": "iShares MSCI India", "region": "India"},
        {"ticker": "EWT", "name": "iShares MSCI Taiwan", "region": "Taiwan"},
        {"ticker": "EWY", "name": "iShares MSCI South Korea", "region": "South Korea"},
    ],
    "global": [
        {"ticker": "VT", "name": "Vanguard Total World Stock", "region": "Global"},
        {"ticker": "ACWI", "name": "iShares MSCI ACWI", "region": "Global"},
        {"ticker": "URTH", "name": "iShares MSCI World", "region": "Developed Global"},
    ],
    "international_bonds": [
        {"ticker": "BNDX", "name": "Vanguard Total International Bond", "region": "Global ex-US"},
        {"ticker": "IAGG", "name": "iShares International Aggregate", "region": "Global ex-US"},
        {"ticker": "EMB", "name": "iShares EM USD Bond", "region": "Emerging Markets"},
        {"ticker": "VWOB", "name": "Vanguard EM Government Bond", "region": "Emerging Markets"},
    ],
}

# All tickers flattened for easy access
def get_all_tickers():
    """Get all available tickers across all universes"""
    all_tickers = set()

    # Stocks
    for category, tickers in STOCK_UNIVERSE.items():
        all_tickers.update(tickers)

    # ETFs
    for category, etfs in ETF_UNIVERSE.items():
        all_tickers.update(e["ticker"] for e in etfs)

    # Bonds
    for category, bonds in BOND_ETF_UNIVERSE.items():
        all_tickers.update(b["ticker"] for b in bonds)

    # Sectors
    for category, sectors in SECTOR_ETF_UNIVERSE.items():
        all_tickers.update(s["ticker"] for s in sectors)

    # Commodities
    for category, commodities in COMMODITY_UNIVERSE.items():
        all_tickers.update(c["ticker"] for c in commodities)

    # REITs
    for category, reits in REIT_UNIVERSE.items():
        all_tickers.update(r["ticker"] for r in reits)

    # International
    for category, intl in INTERNATIONAL_UNIVERSE.items():
        all_tickers.update(i["ticker"] for i in intl)

    return sorted(list(all_tickers))


def get_ticker_info(ticker: str) -> dict:
    """Get information about a specific ticker"""
    # Search through all universes
    for category, tickers in STOCK_UNIVERSE.items():
        if ticker in tickers:
            return {"ticker": ticker, "type": "Stock", "category": category}

    for universe_name, universe in [
        ("ETF", ETF_UNIVERSE),
        ("Bond", BOND_ETF_UNIVERSE),
        ("Sector ETF", SECTOR_ETF_UNIVERSE),
        ("Commodity", COMMODITY_UNIVERSE),
        ("REIT", REIT_UNIVERSE),
        ("International", INTERNATIONAL_UNIVERSE),
    ]:
        for category, items in universe.items():
            for item in items:
                if item["ticker"] == ticker:
                    return {"ticker": ticker, "type": universe_name, "category": category, **item}

    return {"ticker": ticker, "type": "Unknown", "category": "Unknown"}
