clc;clear all;close all;

addpath('npy-matlab/npy-matlab')  
savepath

%PLM_dists = reshape(readNPY('PLM_dists.npy'), [], 3);
%PLM_dists = reshape(readNPY('CNN_preds_normal.npy'), [], 3);

true_locs = reshape(readNPY('bound_true_locs.npy'), [], 2);
PLM_dists = reshape(readNPY('bound_est_dists.npy'), [], 3);

known_references = [7.5, 0; 0, 12.99; 15, 12.99];


mses = 0;
%figure();
%grid on;
%hold on;
for u=1:1
    initial_guess = mean(known_references);
    %true_locs = readNPY('CNN_true_normal.npy');
    est_locs = [];
    for p=1:length(PLM_dists)

        distances = PLM_dists(p,:);




        if size(known_references,2) ~= 2
            error('location of known RPs should be entered as Nx2 matrix');
        end
        % Draw Circles
        %theta = 0:pi/360:2*pi;
        %circle1 = [known_references(1,1)+distances(1)*sin(theta'),known_references(1,2)+distances(1)*cos(theta')];
        %plot(circle1(:,1),circle1(:,2),'b')
        %circle2 = [known_references(2,1)+distances(2)*sin(theta'),known_references(2,2)+distances(2)*cos(theta')];
        %plot(circle2(:,1),circle2(:,2),'b')
        %circle3 = [known_references(3,1)+distances(3)*sin(theta'),known_references(3,2)+distances(3)*cos(theta')];
        %plot(circle3(:,1),circle3(:,2),'b')
        i=1;
        temp_location(i,:) = initial_guess ;
        temp_error = 0 ;
        for j = 1 : size(known_references,1)
            temp_error = temp_error + abs((known_references(j,1) -temp_location(i,1))^2 + (known_references(j,2) - temp_location(i,2))^2 - distances(j)^2) ;
        end
        estimated_error = temp_error ;
        % plot(temp_location(i,1),temp_location(i,2),'k*','MarkerSize', 10) ;
        %plot
        % text(temp_location(i,1), temp_location(i,2)*(1 + 0.8), 'Initial Guess');
        % disp(['The initial location estimation is:', num2str([temp_location(i,1),temp_location(i,2)])]);
        % new matrix = [ ];

        while norm(estimated_error) > 1e-2 %iterative process forLS algorithm
            for j = 1 : size(known_references,1)  %Jacobian has beencalculated in advance
                jacobian_matrix(j,:) = -2*(known_references(j,:) - temp_location(i,:)) ; %partial derivative is i.e. -2(x 1-x)
                f(j) = (known_references(j,1) - temp_location(i,1))^2 +(known_references(j,2) - temp_location(i,2))^2 - distances(j)^2 ;
            end
            estimated_error = -inv(jacobian_matrix' * jacobian_matrix)* (jacobian_matrix') * f' ; %update the U and E
            temp_location(i+1,:) = temp_location(i,:) + estimated_error';
            %plot(temp_location(i+1,1),temp_location(i+1,2),'k*','MarkerSize',10) ;
            % plot
                dp = temp_location(i+1,:)-temp_location(i,:);
            %quiver(temp_location(i,1),temp_location(i,2),dp(1),dp(2),0,'Color','r','LineWidth',2);
                %text(temp location(i+1,1), temp location(i+1,2)*(1 +0.005), num2str(i));
                i = i + 1;
        lx=num2str(temp_location(i,1));ly=num2str(temp_location(i,2));err=sqrt(estimated_error(1)^2+estimated_error(2)^2);
        %disp(['The ',num2str(i-1), 'th estimated location is:','[',lx,',',ly,']',' with an error of ', num2str(err)]);
        end


        for i=1:length(known_references)
            dp = temp_location(end,:) - known_references(i,:);
        %quiver(known_references(i,1),known_references(i,2),dp(1),dp(2),0,'Color','g','LineWidth',2);
        end
        est_locs = [est_locs; [str2double(lx), str2double(ly)]];
        %text(temp_location(end,1), temp_location(end,2)*(1 + 0.2),'Final Estimated Location');
    end

    %plot(est_locs(:,1), est_locs(:,2),'rx', 'MarkerSize',10);
    %plot(true_locs(:,1), true_locs(:,2), 'k+','DisplayName','Truth');
    %plot(known_references(:,1),known_references(:,2),'k^','MarkerSize',10);
    %xlabel('X-location (m)');
    %ylabel('Y-location (m)');
    %legend('Estimates','Truth','Receivers');
    mse = immse(est_locs, true_locs);
    mses = mses + mse;
    %title(strcat('MSE: ', sprintf('%.3f',mse)));
    %xlim([-5, 20]);
    %ylim([-5, 20]);

    writeNPY(true_locs, 'bound_true.npy')
    writeNPY(est_locs, 'bound_ests.npy');
end

mses = mses/1000