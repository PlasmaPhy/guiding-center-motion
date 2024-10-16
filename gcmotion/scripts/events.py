def when_theta(theta0, terminal):

    def event(t, S):
        return S[0] - theta0

    event.terminal = terminal

    return event
