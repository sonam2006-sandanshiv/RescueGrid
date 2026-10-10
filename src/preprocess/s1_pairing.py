import logging
from datetime import datetime
import pystac

logger = logging.getLogger(__name__)

def select_s1_pair(items, flood_date):
    """
    Selects the best S1 pre/post pair.
    The rule: same relative orbit and pass direction, pre <= flood date, post > flood date,
    closest to 12 days apart.
    """
    pre_items = []
    post_items = []
    
    flood_dt = datetime.fromisoformat(flood_date)
    
    for item in items:
        dt = item.datetime
        if dt is None:
            continue
        # timezoneless compare
        dt = dt.replace(tzinfo=None)
        if dt <= flood_dt:
            pre_items.append(item)
        else:
            post_items.append(item)
            
    # Group by relative orbit and pass direction
    best_pair = None
    min_time_diff = None
    
    for pre in pre_items:
        pre_orbit = pre.properties.get("sat:relative_orbit")
        pre_pass = pre.properties.get("sat:orbit_state")
        if pre_orbit is None or pre_pass is None:
            continue
            
        for post in post_items:
            post_orbit = post.properties.get("sat:relative_orbit")
            post_pass = post.properties.get("sat:orbit_state")
            
            if post_orbit == pre_orbit and post_pass == pre_pass:
                time_diff = abs((post.datetime - pre.datetime).total_seconds())
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_pair = (pre, post)
                    
    if best_pair is None:
        logger.warning("Could not find a matching S1 pair with the same orbit.")
        return None
        
    days_apart = min_time_diff / (24 * 3600)
    if not (10 <= days_apart <= 14):
        logger.warning(f"S1 pair is {days_apart:.1f} days apart, fallback from 12 days.")
        
    return best_pair
