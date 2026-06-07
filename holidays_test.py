# https://pypi.org/project/holidays/

import holidays
from datetime import date

# Initialize Russian holidays for 2026
ru_holidays = holidays.Russia(years=[2025, 2026])  # More direct approach

# Check specific dates
print(date(2026, 1, 1) in ru_holidays)  # Should print: True
print(ru_holidays.get(date(2026, 1, 1)))  # Should print holiday name

# Print all holidays in 2026
print("\nAll Russian holidays in 2025-2026:")
for dt, name in sorted(ru_holidays.items()):
    print(f"{dt}: {name}")