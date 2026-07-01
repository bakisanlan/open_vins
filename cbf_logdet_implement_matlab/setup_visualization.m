function [fig, vis] = setup_visualization(params)
%SETUP_VISUALIZATION  Create figure, axes, and all graphics handles.
%
%   [fig, vis] = setup_visualization(params)
%
%   params fields used:
%     N_HIST — log-det history length
%     h_min  — log-det safety threshold
%     pf_d   — feature display coords [3×1]
%
%   Returns:
%     fig — figure handle
%     vis — struct with all plot/text/axes handles

N_HIST = params.N_HIST;
h_min  = params.h_min;
pf_d   = params.pf_d;

%% ── Figure ──────────────────────────────────────────────────────────────
fig = figure('Name','Interactive VIO Observability','Color','k', ...
    'Position',[30 60 1400 780],'MenuBar','none');

%% ── Axes layout ─────────────────────────────────────────────────────────
a3  = axes('Parent',fig,'Position',[0.02 0.05 0.57 0.92]);
ald = axes('Parent',fig,'Position',[0.64 0.55 0.34 0.41]);
att = axes('Parent',fig,'Position',[0.64 0.06 0.34 0.41]);

%% ── 3D scene axes ───────────────────────────────────────────────────────
set(a3,'Color','k','GridColor',[0.3 0.3 0.3],'GridAlpha',0.5, ...
    'XColor',[0.6 0.6 0.6],'YColor',[0.6 0.6 0.6],'ZColor',[0.6 0.6 0.6]);
hold(a3,'on'); grid(a3,'on'); axis(a3,'equal');
xlabel(a3,'East [m]','Color','w'); ylabel(a3,'North [m]','Color','w');
zlabel(a3,'Up [m]','Color','w');
title(a3,'3D Scene — NED→ENU display  |  Red star = feature','Color','w','FontSize',10);
view(a3,35,22);

% Static feature marker
plot3(a3,pf_d(1,:),pf_d(2,:),pf_d(3,:),'p','MarkerSize',18, ...
    'MarkerFaceColor',[1 0.3 0.3],'MarkerEdgeColor','w');

% Dynamic objects (pre-created, updated via set() each tick)
htr  = plot3(a3,nan,nan,nan,'-','Color',[0.45 0.45 0.45],'LineWidth',1);
hbr  = plot3(a3,[nan nan],[nan nan],[nan nan],'Color',[0.2 0.9 0.4],'LineWidth',1.5);
hdr  = plot3(a3,0,0,0,'o','MarkerSize',8,'MarkerFaceColor','w','MarkerEdgeColor','w');
ha1  = plot3(a3,[nan nan],[nan nan],[nan nan],'w-','LineWidth',2.5);
ha2  = plot3(a3,[nan nan],[nan nan],[nan nan],'w-','LineWidth',2.5);
hbX  = plot3(a3,[nan nan],[nan nan],[nan nan],'r-','LineWidth',2);
hbY  = plot3(a3,[nan nan],[nan nan],[nan nan],'g-','LineWidth',2);
hbZ  = plot3(a3,[nan nan],[nan nan],[nan nan],'b-','LineWidth',2);
hVnom  = plot3(a3,[nan nan],[nan nan],[nan nan],'--','Color',[0.6 0.6 0.6],'LineWidth',2);
hVsafe = plot3(a3,[nan nan],[nan nan],[nan nan],'-','Color',[0 1 1],'LineWidth',2.5);
hHUD = text(a3,0.01,0.97,'','Units','normalized','Color','w', ...
    'FontName','Courier','FontSize',8,'VerticalAlignment','top', ...
    'BackgroundColor',[0 0 0 0.65]);
legend(a3,[hbX,hbY,hbZ,hVnom,hVsafe],{'Fwd-X','Right-Y','Down-Z','V_{nom}','V_{safe}'}, ...
    'TextColor','w','Color',[0.15 0.15 0.15],'FontSize',7,'Location','ne');

%% ── Log-det history axes ────────────────────────────────────────────────
set(ald,'Color','k','XColor',[0.7 0.7 0.7],'YColor',[0.7 0.7 0.7]);
hold(ald,'on'); grid(ald,'on');
xlabel(ald,'Steps','Color','w','FontSize',8);
ylabel(ald,'log det(M)','Color','w','FontSize',8);
title(ald,'Observability Monitor','Color','w','FontSize',9,'FontWeight','bold');
yline(ald,h_min,'--','Color',[0.9 0.3 0.3],'LineWidth',1.5, ...
    'Label','h_{min}','FontSize',7,'LabelVerticalAlignment','bottom');
hLD  = plot(ald,1:N_HIST,nan(N_HIST,1),'-','Color',[0.3 0.9 0.5],'LineWidth',1.2);
hLDn = plot(ald,N_HIST,nan,'o','Color',[1 0.85 0.2],'MarkerFaceColor',[1 0.85 0.2],'MarkerSize',7);
hLDt = text(ald,0.98,0.96,'log det = --','Units','normalized', ...
    'HorizontalAlignment','right','Color',[1 0.85 0.2],'FontSize',9,'FontWeight','bold');
xlim(ald,[1,N_HIST]); ylim(ald,[-6 6]);

%% ── Attitude axes ───────────────────────────────────────────────────────
set(att,'Color',[0.05 0.05 0.05],'XColor',[0.5 0.5 0.5], ...
    'YColor',[0.5 0.5 0.5],'ZColor',[0.5 0.5 0.5]);
hold(att,'on'); grid(att,'on'); axis(att,'equal');
xlabel(att,'East','Color','w','FontSize',7);
ylabel(att,'North','Color','w','FontSize',7);
zlabel(att,'Up','Color','w','FontSize',7);
title(att,'Drone Attitude (body axes)','Color','w','FontSize',9,'FontWeight','bold');
view(att,35,20);
xlim(att,[-1.4 1.4]); ylim(att,[-1.4 1.4]); zlim(att,[-1.4 1.4]);

% Reference grid lines
plot3(att,[-1.2 1.2],[0 0],[0 0],'--','Color',[0.35 0.35 0.35]);
plot3(att,[0 0],[-1.2 1.2],[0 0],'--','Color',[0.35 0.35 0.35]);
plot3(att,[0 0],[0 0],[-1.2 1.2],'--','Color',[0.35 0.35 0.35]);

% Horizon fill
[hgx,hgy] = meshgrid(-1.1:0.2:1.1);
surf(att,hgy,hgx,zeros(size(hgx))-0.02,'FaceColor',[0.1 0.25 0.1], ...
    'FaceAlpha',0.35,'EdgeColor','none');

% Attitude body-axis handles
haX  = plot3(att,[0 0],[0 0],[0 0],'r-','LineWidth',3.5);
haY  = plot3(att,[0 0],[0 0],[0 0],'g-','LineWidth',3.5);
haZ  = plot3(att,[0 0],[0 0],[0 0],'b-','LineWidth',3.5);
haA1 = plot3(att,[0 0],[0 0],[0 0],'w-','LineWidth',1.5);
haA2 = plot3(att,[0 0],[0 0],[0 0],'w-','LineWidth',1.5);
hRPY = text(att,0.02,0.05,'R:  0°  P:  0°  Y:  0°', ...
    'Units','normalized','Color','w','FontSize',8,'FontName','Courier');
legend(att,[haX,haY,haZ],{'Forward(X)','Right(Y)','Down(Z)'}, ...
    'TextColor','w','Color',[0.12 0.12 0.12],'FontSize',6.5,'Location','ne');

%% ── Pack all handles ────────────────────────────────────────────────────
vis = struct( ...
    'a3',a3,   'ald',ald,   'att',att, ...
    'htr',htr, 'hdr',hdr,   'hbr',hbr, ...
    'ha1',ha1, 'ha2',ha2, ...
    'hbX',hbX, 'hbY',hbY,   'hbZ',hbZ, ...
    'hVnom',hVnom, 'hVsafe',hVsafe, ...
    'hHUD',hHUD, ...
    'hLD',hLD, 'hLDn',hLDn, 'hLDt',hLDt, ...
    'haX',haX, 'haY',haY,   'haZ',haZ, ...
    'haA1',haA1, 'haA2',haA2, 'hRPY',hRPY);

end
