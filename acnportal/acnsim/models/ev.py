from builtins import property


class EV:

    """Class to model the behavior of an Electrical Vehicle (ev).

    Args:
        arrival (int): Arrival time of the ev. [periods]
        departure (int): Departure time of the ev. [periods]
        requested_energy (float): Energy requested by the ev on arrival.
            [kWh]
        station_id (str): Identifier of the station used by this ev.
        session_id (str): Identifier of the session belonging to this 
            ev.
        battery (Battery-like): Battery object to be used by the EV.
    """
    def __init__(self, arrival, departure, requested_energy, station_id,
                 session_id, battery, estimated_departure=None):
        # User Defined Parameters
        self._arrival = arrival
        self._departure = departure
        self._session_id = session_id
        self._station_id = station_id

        # Estimate of session parameters
        self._requested_energy = requested_energy
        self._estimated_departure = (estimated_departure
                                     if estimated_departure is not None
                                     else departure)
        # Internal State
        self._battery = battery
        self._energy_delivered = 0
        self._current_charging_rate = 0

    def __eq__(self, other):
        # TODO: Is this an acceptable definition?
        return self.session_id == other.session_id

    @property
    def arrival(self):
        """ Return the arrival time of the EV."""
        return self._arrival

    @arrival.setter
    def arrival(self, value):
        """ Set the arrival time of the EV. (int) """
        self._arrival = value

    @property
    def departure(self):
        """ Return the departure time of the EV. (int) """
        return self._departure

    @departure.setter
    def departure(self, value):
        """ Set the departure time of the EV. (int) """
        self._departure = value

    @property
    def estimated_departure(self):
        """ Return the estimated departure time of the EV."""
        return self._estimated_departure

    @estimated_departure.setter
    def estimated_departure(self, value):
        """ Set the estimated departure time of the EV. (int) """
        self._estimated_departure = value

    @property
    def requested_energy(self):
        """ Return the energy request of the EV for
            this session. (float) [acnsim units]. """
        return self._requested_energy

    @property
    def session_id(self):
        """ Return the unique session identifier for this charging
        session. (str) """
        return self._session_id

    @property
    def station_id(self):
        """ Return the unique identifier for the EVSE used for this
        charging session. """
        return self._station_id

    @property
    def energy_delivered(self):
        """ Return the total energy delivered so far in this charging
        session. (float) """
        return self._energy_delivered

    @property
    def current_charging_rate(self):
        """ Return the current charging rate of the EV. (float) """
        return self._current_charging_rate

    @property
    def remaining_demand(self):
        """ Return the remaining energy demand of this session. (float)

        Defined as the difference between the requested energy of the
        session and the energy delivered so far.
        """
        return self.requested_energy - self.energy_delivered

    @property
    def fully_charged(self):
        """ Return True if the EV's demand has been fully met. (bool)"""
        return not (self.remaining_demand > 1e-3)

    @property
    def percent_remaining(self):
        """ Return the percent of demand which still needs to be
        fulfilled. (float)

        Defined as the ratio of remaining demand and requested
        energy. """
        return self.remaining_demand / self.requested_energy

    @property
    def maximum_charging_power(self):
        """ Return the maximum charging power of the battery."""
        return self._battery.max_charging_power

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the ev.

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery charger.
                [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: Actual charging rate of the ev. [A]
        """
        charge_rate = self._battery.charge(pilot, voltage, period)
        self._energy_delivered += ((charge_rate * voltage)
                                   / 1000
                                   * (period / 60))
        self._current_charging_rate = charge_rate
        return charge_rate

    def reset(self):
        """ Reset battery back to its initialization. Also reset energy
        delivered.

        Returns:
            None.
        """
        self._energy_delivered = 0
        self._battery.reset()
