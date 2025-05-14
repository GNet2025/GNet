clear all;
clc;
close all;

load('Epilepsy_HDCGNet.mat')
load('Epilepsy_HoloGN.mat')
load('Epilepsy_LaplaceHDC.mat')
load('Epilepsy_nHDC.mat')
load('Epilepsy_OnlineHD.mat')
load('Epilepsy_RFFHDC.mat')
load('Epilepsy_VanillaHDC.mat')
GNetAcc = 93.48;

x = mean(Epilepsy_nHDC,2);

m0 = GNetAcc*ones(11,1);
m1 = mean(Epilepsy_HDCGNet, 2);
m2 = mean(Epilepsy_HoloGN,2);
m3 = mean(Epilepsy_LaplaceHDC,2);
m4 = mean(Epilepsy_OnlineHD,2);
m5 = mean(Epilepsy_RFFHDC,2);
m6 = mean(Epilepsy_VanillaHDC,2);

s1 = std(Epilepsy_HDCGNet,0,2);
s2 = std(Epilepsy_HoloGN,0,2);
s3 = std(Epilepsy_LaplaceHDC,0,2);
s4 = std(Epilepsy_OnlineHD,0,2);
s5 = std(Epilepsy_RFFHDC,0,2);
s6 = std(Epilepsy_VanillaHDC,0,2);

gamma = .5;

m1u = m1 + gamma*s1;
m1l = m1 - gamma*s1;
m2u = m2 + gamma*s2;
m2l = m2 - gamma*s2;
m3u = m3 + gamma*s3;
m3l = m3 - gamma*s3;
m4u = m4 + gamma*s4;
m4l = m4 - gamma*s4;
m5u = m5 + gamma*s5;
m5l = m5 - gamma*s5;
m6u = m6 + gamma*s6;
m6l = m6 - gamma*s6;

xconf = [x' x(end:-1:1)'] ;
yconf1 = [m1u' m1l(end:-1:1)'];
yconf2 = [m2u' m2l(end:-1:1)'];
yconf3 = [m3u' m3l(end:-1:1)'];
yconf4 = [m4u' m4l(end:-1:1)'];
yconf5 = [m5u' m5l(end:-1:1)'];
yconf6 = [m6u' m6l(end:-1:1)'];


% Create figure
figure1 = figure('position',[100 100 800 700]);

% Create axes
axes1 = axes('Parent',figure1,...
    'Position',[0.088 0.11 0.86 0.84]);
hold(axes1,'on');


% Create patch
patch('DisplayName','conf1','Parent',axes1,'YData',yconf1,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.85 0.51 0.34],...
    'EdgeColor','none');

% Create patch
patch('DisplayName','conf2','Parent',axes1,'YData',yconf2,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.6 0.30 0.11],...
    'EdgeColor','none');

% Create patch
patch('DisplayName','conf3','Parent',axes1,'YData',yconf3,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.29 0.73 0.87],...
    'EdgeColor','none');

% Create patch
patch('DisplayName','conf4','Parent',axes1,'YData',yconf4,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.28 0.83 0.37],...
    'EdgeColor','none');

% Create patch
patch('DisplayName','conf5','Parent',axes1,'YData',yconf5,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.97 0.77 0.34],...
    'EdgeColor','none');

% Create patch
patch('DisplayName','conf6','Parent',axes1,'YData',yconf6,...
    'XData',xconf,...
    'FaceAlpha',0.5,...
    'FaceColor',[0.82 0.64 0.95],...
    'EdgeColor','none');

% Create multiple line objects using matrix input to plot
plot1 = plot(x,[m0,m1,m2,m3,m4,m5,m6],'SeriesIndex',3,'MarkerSize',9,'LineWidth',3,...
    'Parent',axes1);
set(plot1(1),'DisplayName','GNet','LineStyle','--',...
    'Color',[0 0 0]);
set(plot1(2),'DisplayName','HDC-GNet','Marker','o',...
    'Color',[0.85 0.29 0.051]);
set(plot1(3),'DisplayName','Holo-GN','Marker','^',...
    'Color',[0.53 0.25 0.06]);
set(plot1(4),'DisplayName','Laplace-HDC','Marker','*',...
    'Color',[0.21 0.59 0.90]);
set(plot1(5),'DisplayName','OnlineHD','Marker','+',...
    'Color',[0.19 0.77 0.13]);
set(plot1(6),'DisplayName','RFF-HDC','Marker','square',...
    'Color',[0.91 0.53 0.063]);
set(plot1(7),'DisplayName','Classic HDC','Marker','pentagram',...
    'Color',[0.58 0.3 0.93]);


% Create ylabel
ylabel('Accuracy (\%)','Interpreter','latex');

% Create xlabel
xlabel('Hyperdimension ($N$)','Interpreter','latex');


xlim([min(x),max(x)])

box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'FontSize',20,'XGrid','on','YGrid','on','YMinorTick','on');
%Create legend
legend1 = legend(plot1,{'GNet-Baseline', 'HDC-GNet' , 'HoloGN', ...
    'LaplaceHDC' , 'OnlineHD', 'RFF-HDC', 'Classic-HDC'});
set(legend1,...
    'Position',[0.592918081037547 0.126788415528288 0.349804305283757 0.193679092382496],...
    'FontSize',24,'NumColumns',2, 'Interpreter','latex');
