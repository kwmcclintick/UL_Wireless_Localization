% Generated by MATLAB(R) 9.8 (R2020a) and WLAN Toolbox 3.0 (R2020a).
% Generated on: 24-Jul-2020 14:03:58

%% Generate 802.11b/g (DSSS) Waveform
% 802.11b/g (DSSS) configuration:
dsssCfg = wlanNonHTConfig('Modulation', 'DSSS', ...
    'DataRate', '1Mbps', ...
    'Preamble', 'Long', ...
    'LockedClocks', true, ...
    'PSDULength', 1000);

numPackets = 1;
% input bit source:
in = randi([0, 1], 1000, 1);


% waveform generation:
waveform = wlanWaveformGenerator(in, dsssCfg, ...
    'NumPackets', 1, ...
    'IdleTime', 0);

Fs = wlanSampleRate(dsssCfg); 								 % sample rate of waveform

%% Impair 802.11b/g (DSSS) Waveform
% AWGN
waveform = awgn(waveform, 20, 'measured');

%% Visualize 802.11b/g (DSSS) Waveform
% Time Scope
timeScope = dsp.TimeScope('SampleRate', Fs, ...
    'TimeSpanOverrunAction', 'Scroll', ...
    'TimeSpan', 2.7273e-06);
timeScope(waveform);
release(timeScope);

% Spectrum Analyzer
spectrum = dsp.SpectrumAnalyzer('SampleRate', Fs);
spectrum(waveform);
release(spectrum);

% Constellation Diagram
constel = comm.ConstellationDiagram('ColorFading', true, ...
    'ShowTrajectory', 0, ...
    'ShowReferenceConstellation', false);
constel(waveform);
release(constel);


