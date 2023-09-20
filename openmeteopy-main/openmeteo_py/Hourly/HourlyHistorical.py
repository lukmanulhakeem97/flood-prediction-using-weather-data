from openmeteo_py.Exceptions import *



class HourlyHistorical()  :

    """
    Hourly Parameter functions

    Hourly Parameter Definition
    Most weather variables are given as an instantaneous value for the indicated hour. 
    Some variables like precipitation are calculated from the preceding hour as and average or sum.

    """


    def __init__(self) :
        self.hourly_params = TypedList()

    def temperature_2m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Air temperature at 2 meters above ground
        """

        self.hourly_params.append("temperature_2m")
        return self

    def relativehumidity_2m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`
        
        Relative humidity at 2 meters above ground
        
        """
        self.hourly_params.append("relativehumidity_2m")
        return self

    def dewpoint_2m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Dew point temperature at 2 meters above ground
        
        """

        self.hourly_params.append("dewpoint_2m")
        return self

    def apparent_temperature(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Apparent temperature is the perceived feels-like tempertature combinding wind chill factor, realtive humidity and solar radition

        
        """

        self.hourly_params.append("apparent_temperature")
        return self

    def pressure_msl(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Atmospheric air pressure reduced to sea level
        
        """

        self.hourly_params.append("pressure_msl")
        return self

    def cloudcover(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Total cloud cover as an area fraction
        
        """

        self.hourly_params.append("cloudcover")
        return self

    def cloudcover_low(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Low level clouds and fog up to 3 km altitude
        
        """

        self.hourly_params.append("cloudcover_low")
        return self

    def cloudcover_mid(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Mid level clouds from 3 to 8 km altitude

        
        """

        self.hourly_params.append("cloudcover_mid")
        return self

    def cloudcover_high(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        High level clouds from 8 km altitude
        
        """

        self.hourly_params.append("cloudcover_high")
        return self

    def windspeed_10m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level.

        
        """

        self.hourly_params.append("windspeed_10m")
        return self
    def windspeed_100m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Wind speed at 10 meters above ground. Wind speed on 10 meters is the standard level.

        
        """

        self.hourly_params.append("windspeed_80m")
        return self

    def winddirection_10m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Wind direction at 10 meters above ground
        """

        self.hourly_params.append("winddirection_10m")
        return self
    
    def winddirection_100m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Wind direction at 10 meters above ground
        """

        self.hourly_params.append("winddirection_80m")
        return self

    def windgusts_10m(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Gusts at 10 meters above ground as a maximum of the preceding hour
        """

        self.hourly_params.append("windgusts_10m")
        return self

    def shortwave_radiation(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Shortwave solar radiation as average of the preceding hour
        """

        self.hourly_params.append("shortwave_radiation")
        return self

    def direct_radiation(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Direct solar radiation as average of the preceding hour
        """

        self.hourly_params.append("irect_radiation")
        return self

    def diffuse_radiation(self):
        """
        Returns the Hourly configuration object
        :returns: `Hourly()`

        Diffure solar radiation as average of the preceding hour
        """

        self.hourly_params.append("diffuse_radiation")
        return self

    def vapor_pressure_deficit(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Vapor Pressure Deificit (VPD) in kilo pascal (kPa). For high VPD (>1.6), water transpiration of plants increases. For low VPD (<0.4), transpiration decreases
        """

        self.hourly_params.append("vapor_pressure_deficit")
        return self

    def precipitation(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Total precipitation (rain, showers, snow) sum of the preceding hour
        """

        self.hourly_params.append("precipitation")
        return self
    
    def precipitation_probability(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Total precipitation (rain, showers, snow) sum of the preceding hour
        """

        self.hourly_params.append("precipitation_probability")
        return self

    def weathercode(self):
        """
        Returns the Hourly configuration object
        :returns: `Hourly()`

        Weather condition as a numeric code. Follow WMO weather interpretation codes. See table below for details.
        """

        self.hourly_params.append("weathercode")
        return self

    def snowfall(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7.
        """
        self.hourly_params.append("snowfall")
        return self
    
    def snow_depth(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Snowfall amount of the preceding hour in centimeters. For the water equivalent in millimeter, divide by 7.
        """
        self.hourly_params.append("snow_depth")
        return self
    
    def direct_normal_irradiance(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Direct solar radiation as average of the preceding hour on the horizontal plane and the normal plane.
        """
        self.hourly_params.append("direct_normal_irradiance")
        return self
    
    def cape(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Convective available potential energy.
        """
        self.hourly_params.append("cape")
        return self
    
    def et0_fao_evapotranspiration(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("et0_fao_evapotranspiration")
        return self
    def soil_moisture_28_to_100cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_moisture_28_to_100cm")
        return self
    def soil_temperature_0_to_7cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_temperature_0_to_7cm")
        return self
    def soil_temperature_7_to_28cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_temperature_7_to_28cm")
        return self
    def soil_temperature_28_to_100cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_temperature_28_to_100cm")
        return self
    def soil_temperature_100_to_255cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_temperature_100_to_255cm")
        return self
    def soil_moisture_0_to_7cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_moisture_0_to_7cm")
        return self
    def soil_moisture_7_to_28cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        ET₀ Reference Evapotranspiration of a well watered grass field. Based on FAO-56 Penman-Monteith equations ET₀ is calculated from temperature, wind speed, humidity and solar radiation. Unlimited soil water is assumed. ET₀ is commonly used to estimate the required irrigation for plants.
        """
        self.hourly_params.append("soil_moisture_7_to_28cm")
        return self
    
    def surface_pressure(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        Atmospheric air pressure reduced to mean sea level (msl) or pressure at surface. Typically pressure on mean sea level is used in meteorology. Surface pressure gets lower with increasing elevation..
        """
        self.hourly_params.append("surface_pressure")
        return self
    
    def soil_moisture_100_to_255cm(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()` 
        geopotential_height at 800hPa
        """
        self.hourly_params.append("soil_moisture_100_to_200cm")
        return self

    def all(self):
        """
        Returns the Hourly configuration object 
        :returns: `Hourly()`

        All hourly parameters
        """
        self.hourly_params.append_all(
            ["temperature_2m",
            "relativehumidity_2m",
            "dewpoint_2m",
            "apparent_temperature",
            "pressure_msl",
            "surface_pressure",
            "cloudcover",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "windspeed_10m",
            "windspeed_100m",
            "winddirection_10m",
            "winddirection_100m",
            "windgusts_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "vapor_pressure_deficit",
            "et0_fao_evapotranspiration",
            "precipitation",
            "weathercode",
            "snowfall",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
            "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm",
            "soil_temperature_0_to_7cm",
            "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm",
            "soil_temperature_100_to_255cm"
            ])
        return self
