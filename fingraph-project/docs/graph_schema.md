# FinGraph Schema Design

## Node Types

### 1. Company Nodes
**Purpose**: Individual publicly traded companies
**Features** (20+ features):
- Financial: P/E ratio, ROE, debt-to-equity, revenue growth
- Technical: RSI, MACD, volatility, beta
- Market: Market cap, sector, geographic region
- Sentiment: News sentiment score, social media mentions

### 2. Sector Nodes  
**Purpose**: Industry groupings (Technology, Healthcare, etc.)
**Features**:
- Aggregate financial metrics
- Sector performance vs market
- Economic sensitivity scores
- Regulatory environment indicators

### 3. Economic Indicator Nodes
**Purpose**: Macroeconomic factors
**Features**:
- Interest rates (Fed funds rate, 10-year treasury)
- Economic growth (GDP, employment)
- Inflation measures (CPI, PPI)
- Market indices (S&P 500, VIX)

### 4. Geographic Nodes
**Purpose**: Regional economic factors
**Features**:
- Regional GDP growth
- Local economic indicators
- Currency exchange rates
- Political stability scores

## Edge Types

### 1. Correlation Edges (Company ↔ Company)
**Purpose**: Statistical relationships between stock prices
**Features**:
- 30/60/90-day rolling correlation
- Volatility correlation
- Trading volume correlation
- Directional causality measures

### 2. Supply Chain Edges (Company → Company)
**Purpose**: Business relationships and dependencies
**Features**:
- Supplier-customer relationships (from SEC filings)
- Relationship strength (revenue percentage)
- Geographic proximity
- Industry interdependencies

### 3. Sector Membership Edges (Company → Sector)
**Purpose**: Industry classification
**Features**:
- Primary/secondary sector membership
- Revenue allocation by sector
- Sector beta correlation
- Competitive positioning

### 4. Economic Impact Edges (Economic Indicator → Company/Sector)
**Purpose**: How macro factors affect specific entities
**Features**:
- Interest rate sensitivity
- Economic cycle correlation
- Currency exposure
- Regulatory impact scores

## Graph Statistics (Target)
- **Nodes**: ~1,000 (500 companies + 50 sectors + 20 indicators + geographic)
- **Edges**: ~5,000-10,000 (dense network for good GNN performance)
- **Average Degree**: 10-20 (well-connected but not overly dense)
- **Features per Node**: 15-30 (rich but manageable)