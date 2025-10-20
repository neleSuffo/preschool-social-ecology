def time_to_seconds(time_str):
    """Converts MM:SS or float seconds string to float seconds.
    
    Parameters:
    ----------
    time_str : str
        Time in MM:SS format or as a float string.
        
    Returns:
    -------
    float or None
        Time in seconds as a float, or None if conversion fails.
    """
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return None