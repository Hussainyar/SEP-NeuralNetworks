!===============================================================================
! SPIKING NEURAL NETWORK SIMULATION
! Purpose: Simulate a network of integrate-and-fire neurons with synaptic plasticity
! Language: Fortran 90
!===============================================================================

!===============================================================================
! PARAMETER DEFINITIONS
!===============================================================================

	parameter(ntrail=1)
	parameter(npd=1)
	parameter(nn=3000,max=200000)
	
	! Population fractions
	parameter(p=0.1d0)
	parameter(pe1=0.80,pe2=0.80-pe1)
	parameter(ne1=nn*pe1,ne2=nn*pe2,ne=ne1+ne2)
	
	! Network connectivity
	parameter(nnei=nn*p*5)
	
	! Simulation time step
	parameter(h=0.05d0)
	
	! Refractory period
	parameter(nrp0=50)
	
	! Time constants (ms)
	parameter(tex=15d0,tinh=5d0,tinhf=0d0)
	
	! Firing timing parameters
	Parameter(deltatau=0d0,tauex1=10d0-deltatau)
	parameter(tauex2=10d0+deltatau,tauin=20d0)
	
	! Membrane potential and reversal potentials (mV)
	parameter(vrest=-60d0,vth=-50d0,eex=0d0,einh=-80d0)
	
	! Threshold variability
	parameter(vthb=0d0,vthf=0d0,vthd=0d0)
	
	! Input parameters
	parameter(theinput=0.3d0)
	parameter(ninput=5)
	
	! Binning for analysis (10ms bins)
	parameter(n_bin=200)
	
	! Neuron indices to track
	parameter(itrack1=194,itrack2=159,itrack3=142)
	parameter(itrack4=600,itrack5=430,itrack6=486)


!===============================================================================
! VARIABLE DECLARATIONS
!===============================================================================

	! Neuron position arrays
	real*8 position_ne1(ne1,2)
	real*8 xposition
	integer norder(ne1),norder1(ne1)
	
	! Tracking and analysis variables
	integer iif(3),iifd(3,n_bin),iifm(3),iik,iiik
	
	! Synaptic conductance increments
	real*8 dgexe,dgexi,dginhe,dginhi
	
	! Time constants and exponential decay factors
	real*8 etex,etinh(nn)
	
	! Neuronal state variables
	real*8 v(nn)                    ! Membrane potential
	real*8 vthi(nn),vthi0           ! Action potential threshold
	real*8 gex(nn),ginh(nn)         ! Excitatory and inhibitory conductances
	real*8 a(nn),b(nn)              ! Voltage equation coefficients
	
	! Refractory period and connectivity
	integer ninput0,n_inputs_start
	integer nrp(nn),fire(nn),dr(nn,nnei)
	
	! Random number generator seeds
	integer idum,ix1,ix2,ix3
	integer iduma,ix1a,ix2a,ix3a
	integer idumb,ix1b,ix2b,ix3b
	
	! Temporary variables for computation
	real*8 x1,xinput,xselect,xninput,xv,ex1
	
	! Firing rate storage
	real*8 fr(ntrail,npd,nn)
	
	! Current and synaptic current tracking
	real*8 currentdif(6),cfp(6),cfn(6)
	real*8 scfp(6),scfn(6),acfp(6),acfn(6)
	real*8 tinhi(nn)


!===============================================================================
! MAIN PROGRAM EXECUTION
!===============================================================================

	! Output file setup (commented out)
	!open(1,file='firing_rate_sta_test.dat')


!===============================================================================
! SECTION 1: LOAD AND ORDER NEURON POSITIONS
!===============================================================================

	! Read neuron positions from file
	open(75,file='Position_N3000.dat')
	do i=1,ne1
		read(75,175) position_ne1(i,1),position_ne1(i,2)
	enddo
175	format(1x,2e17.8)
	close(75)

	! Initialize ordering arrays
	norder1=0
	norder=0
	
	! Order neurons by position (first 1000)
	!do i=1,1000
	do i=1, nn/5
		xposition=position_ne1(i,1)
		!do j=1,1000
		do j=1,nn/5
			if(position_ne1(j,1)<=xposition) then
				norder1(i)=norder1(i)+1
			endif
		enddo
	enddo
	
	! Set remaining neurons
	!do i=1001,ne1
	do i=(nn/5) +1,ne1
		norder1(i)=nn/5
	enddo

	! Order neurons by position (remaining neurons)
	!do i=1001,ne1
	do i=(nn/5) +1,ne1
		xposition=position_ne1(i,1)
		!do j=1001,ne1
		do j=(nn/5) +1,ne1
			if(position_ne1(j,1)<=xposition) then
				norder1(i)=norder1(i)+1
			endif
		enddo
	enddo

	! Create final ordering map
	do i=1,ne1
		norder(norder1(i))=i
	enddo


!===============================================================================
! SECTION 2: INITIALIZE FIRING RATES
!===============================================================================

	do itrail=1,ntrail
		do ipd=1,npd
			do iii=1,nn
				fr(itrail,ipd,iii)=0d0
			enddo
		enddo
	enddo


!===============================================================================
! SECTION 3: MAIN SIMULATION LOOP - PARAMETER VARIATIONS
!===============================================================================

	do 2000 ipd=1,npd
		
		! Conductance parameters
		dgexstep=0.0d0*ipd
		dgexe=0.2d0+0.00d0        ! Excitatory to excitatory
		dgexi=0.45d0+0.00d0       ! Excitatory to inhibitory
		dginhe=3.05d0              ! Inhibitory to excitatory
		dginhi=3d0                 ! Inhibitory to inhibitory

		! Compute exponential decay factor for excitatory synapses
		etex=exp(-h/tex)
		
		! Initialize inhibitory time constants with variability
		do ii=1,nn
			call ran2(idum,ex1,ix1,ix2,ix3)	
			tinhi(ii)=tinh+tinhf*(ex1-0.5d0)
			etinh(ii)=exp(-h/tinhi(ii))
		enddo

		! Reset random seed
		idum=-1

		! Initialize thresholds with variability
		do ithi=1,nn
			call ran2(idum,x1,ix1,ix2,ix3)
			vthi(ithi)=vth+vthb*(x1-0.5)
		enddo

		! Apply threshold adjustment to specific neuron population
		do iii=(nn/5)+1,(nn*0.32)
			vthi(norder(iii))=vth+vthd
		enddo


!===============================================================================
! SECTION 4: MAIN SIMULATION LOOP - TRIALS
!===============================================================================

	do 1000 itrail=1,ntrail

		! Set input range
		n_inputs_range=ne
		n_inputs_start=1

		write(*,*) 'p1f',n_inputs_start,n_inputs_range
		write(*,*) itrail

		! Open connectivity file
		open(2,file='ConnectionN3000_100-700.dat')

		! Open output files
		open(73,file='firing_rates_N3000.dat')
		open(78,file='firing_pattern_N3000.dat')

!===============================================================================
! SECTION 5: LOAD CONNECTION MATRIX
!===============================================================================

		do ii=1,nn
			do jj=1,nnei
				read(2,*) it,jt,dr(ii,jj)
			enddo
		enddo
22		format(1x,3i10)

!===============================================================================
! SECTION 6: INITIALIZE NEURONAL STATE VARIABLES
!===============================================================================

		iduma=-5-itrail
		idumb=-4-itrail
		
		do ii=1,nn
			! Initialize membrane potentials
			call ran2(idum,x1,ix1,ix2,ix3)
			v(ii)=vrest+exp((x1-1)*100d0)*(vth-vrest)
			
			! Initialize excitatory conductances
			call ran2(idum,x1,ix1,ix2,ix3)
			gex(ii)=exp((x1-1)*100d0)*0.1d0
			
			! Initialize inhibitory conductances
			call ran2(idum,x1,ix1,ix2,ix3)
			ginh(ii)=exp((x1-1)*100d0)*nnei*(dginhe+dginhi)*0.5d0
		enddo

		! Initialize time
		t=0d0

		! Initialize current tracking
		scfp=0d0
		scfn=0d0
		acfp=0d0
		acfn=0d0
		iifd=0
		iifm=0
		
		do iiik=1,n_bin
			iifm(:)=iifm(:)+iifd(:,iiik)
		enddo
		iik=1


!===============================================================================
! SECTION 7: MAIN TIME STEPPING LOOP
!===============================================================================

	do 10 i=1,max
	
		! Progress marker
		if(i/10000*10000.eq.i) then
			write(*,*) i
		endif

		! Compute current time
		t=h*i

!===============================================================================
! SECTION 7.1: TRACK CURRENTS (after 10% of simulation)
!===============================================================================

		if(i.gt.(max/10)) then
			
			! Compute excitatory and inhibitory currents for tracked neurons
			cfp(1)=gex(itrack1)*(eex-(v(itrack1)))
			cfn(1)=ginh(itrack1)*(einh-(v(itrack1)))
			cfp(2)=gex(itrack2)*(eex-(v(itrack2)))
			cfn(2)=ginh(itrack2)*(einh-(v(itrack2)))
			cfp(3)=gex(itrack3)*(eex-(v(itrack3)))
			cfn(3)=ginh(itrack3)*(einh-(v(itrack3)))
			cfp(4)=gex(itrack4)*(eex-(v(itrack4)))
			cfn(4)=ginh(itrack4)*(einh-(v(itrack4)))
			cfp(5)=gex(itrack5)*(eex-(v(itrack5)))
			cfn(5)=ginh(itrack5)*(einh-(v(itrack5)))
			cfp(6)=gex(itrack6)*(eex-(v(itrack6)))
			cfn(6)=ginh(itrack6)*(einh-(v(itrack6)))

			if((i/100*100).eq.i) then
				! Output current data (commented out)
				! write(77,177) cfp(1),cfn(1),v(itrack1),cfp(2),cfn(2),v(itrack2)
177				format(1x,6e17.8)
			endif

			! Accumulate currents for averaging
			do ipn=1,6
				scfp(ipn)=scfp(ipn)+cfp(ipn)
				scfn(ipn)=scfn(ipn)+cfn(ipn)
			enddo

		endif

		! Initialize spike counter
		iif=0


!===============================================================================
! SECTION 7.2: UPDATE EACH NEURON
!===============================================================================

		do ii=1,nn

!===============================================================================
! SECTION 7.2.1: PROCESS INCOMING SPIKES (SYNAPTIC TRANSMISSION)
!===============================================================================

			do jj=1,nnei
				jt=dr(ii,jj)
				
				! Skip if no connection
				if(jt.eq.0) then
					goto 100
				endif
				
				! If presynaptic neuron fired
				if(fire(jt).eq.1) then 
					
					! Excitatory presynaptic neuron
					if(jt.le.ne) then
						! Excitatory postsynaptic neuron
						if(ii.le.ne) then
							gex(ii)=gex(ii)+dgexe
						! Inhibitory postsynaptic neuron
						elseif(ii.gt.ne) then
							gex(ii)=gex(ii)+dgexi
						endif
					
					! Inhibitory presynaptic neuron
					elseif(jt.gt.ne) then
						! Excitatory postsynaptic neuron
						if(ii.le.ne) then
							ginh(ii)=ginh(ii)+dginhe
						! Inhibitory postsynaptic neuron
						elseif(ii.gt.ne) then
							ginh(ii)=ginh(ii)+dginhi
						endif
					endif
				endif
			enddo
100			continue

!===============================================================================
! SECTION 7.2.2: SYNAPTIC DECAY
!===============================================================================

			gex(ii)=gex(ii)*etex
			ginh(ii)=ginh(ii)*etinh(ii)

!===============================================================================
! SECTION 7.2.3: UPDATE MEMBRANE POTENTIAL (IF NOT IN REFRACTORY PERIOD)
!===============================================================================

			if(nrp(ii).eq.0) then
				
				! Population 1 excitatory neurons
				if(ii.le.ne1) then
					a(ii)=(vrest+gex(ii)*eex+ginh(ii)*einh)/tauex1
					b(ii)=(1d0+gex(ii)+ginh(ii))/tauex1
					v(ii)=v(ii)*exp(-b(ii)*h)+(1d0-exp(-b(ii)*h))*a(ii)/b(ii)
				
				! Population 2 excitatory neurons
				elseif(ii.gt.ne1.and.ii.le.ne)then
					a(ii)=(vrest+gex(ii)*eex+ginh(ii)*einh)/tauex2
					b(ii)=(1d0+gex(ii)+ginh(ii))/tauex2
					v(ii)=v(ii)*exp(-b(ii)*h)+(1d0-exp(-b(ii)*h))*a(ii)/b(ii)
				
				! Inhibitory neurons
				elseif(ii.gt.ne)then
					a(ii)=(vrest+gex(ii)*eex+ginh(ii)*einh)/tauin	
					b(ii)=(1d0+gex(ii)+ginh(ii))/tauin
					v(ii)=v(ii)*exp(-b(ii)*h)+(1d0-exp(-b(ii)*h))*a(ii)/b(ii)
				endif
			endif
	
!===============================================================================
! SECTION 7.2.4: CHECK FOR SPIKE (ACTION POTENTIAL)
!===============================================================================

			call ran2(idum,xv,ix1,ix2,ix3)
			vthi0=vthi(ii)+(xv-0.5d0)*vthf
			
			if(v(ii).ge.vthi0) then
				
				! Reset potential to resting value
				v(ii)=vrest
				
				! Enter refractory period
				nrp(ii)=nrp0
				
				! Record spike timing
				if(i.gt.(max/5)) then
					write(78,178) t,ii
				endif
178				format(1x,e17.8,i5)

				! Count spikes
				iif(1)=iif(1)+1
				
				! Categorize by population
				if(ii.le.ne) then
					iif(2)=iif(2)+1
				elseif(ii.gt.ne) then
					iif(3)=iif(3)+1
				endif
			endif

		enddo ! End neuron loop


!===============================================================================
! SECTION 7.3: UPDATE SPIKE COUNTS FOR BINNED ANALYSIS
!===============================================================================

		! Update moving window of spike counts
		iifm(:)=iifm(:)-iifd(:,iik)
		iifm(:)=iifm(:)+iif(:)
		iifd(:,iik)=iif(:)
		
		! Move to next bin
		iik=iik+1
		if(iik.gt.n_bin) then
			iik=iik-n_bin
		endif

		! Output firing rates at intervals
		if((i/30*30).eq.i) then
			write(73,173) t,iif,iifm
		endif
173		format(1x,e17.8,6i5)

!===============================================================================
! SECTION 7.4: EXTERNAL INPUT
!===============================================================================

		call ran2(iduma,xinput,ix1a,ix2a,ix3a)

		! Apply external excitatory input with probability theinput
		if(xinput.lt.theinput) then

			call ran2(iduma,xninput,ix1a,ix2a,ix3a)
			ninput0=int(ninput*xninput)
			
			! Add input to randomly selected neurons
			do iinput=1,ninput0
				call ran2(idumb,xselect,ix1b,ix2b,ix3b)
				jt=int((xselect-1d-19)*n_inputs_range)

				! Increase excitatory conductance
				gex(norder(jt))=gex(norder(jt))+dgexe
			enddo

		endif

!===============================================================================
! SECTION 7.5: UPDATE FIRE FLAGS AND REFRACTORY PERIODS
!===============================================================================

		fire=0
		do ii=1,nn
			! Mark neuron as firing if it just fired
			if(nrp(ii).eq.nrp0) then
				fire(ii)=1

				! Update firing rate statistics (after 20% of simulation)
				if(i.gt.(max/5)) then
					fr(itrail,ipd,ii)=fr(itrail,ipd,ii)+1d0
				endif
			endif
			
			! Decrement refractory period counter
			if(nrp(ii).ne.0) then
				nrp(ii)=nrp(ii)-1
			endif
		enddo

10	continue ! End main time stepping loop

	close(2)

!===============================================================================
! SECTION 8: COMPUTE AND OUTPUT FIRING RATES
!===============================================================================

		do iii=1,nn
			! Convert spike count to firing rate (Hz)
			fr(itrail,ipd,iii)=fr(itrail,ipd,iii)*1000*5/(5*max*h)
			
			! Output firing rates
			write(1,101) itrail,ipd,fr(itrail,ipd,iii)
		enddo
101		format(1x,2i4,e17.8)

!===============================================================================
! SECTION 9: COMPUTE AND OUTPUT AVERAGED CURRENTS
!===============================================================================

		do ipn=1,6
			! Average currents over recording period
			acfp(ipn)=scfp(ipn)/(max-(max/5))
			acfn(ipn)=scfn(ipn)/(max-(max/5))
			currentdif(ipn)=(acfp(ipn)+acfn(ipn))
		enddo

		! Output current statistics for tracked neurons
		write(*,710) currentdif(1),acfp(1),acfn(1)
		write(*,710) currentdif(2),acfp(2),acfn(2)
		write(*,710) currentdif(3),acfp(3),acfn(3)
		write(*,710) currentdif(4),acfp(4),acfn(4)
		write(*,710) currentdif(5),acfp(5),acfn(5)
		write(*,710) currentdif(6),acfp(6),acfn(6)
710		format(1x,3e17.8)

1000	continue ! End trial loop
2000	continue ! End parameter variation loop
	
	! Close output files
	close(1)
	close(73)

	end


!===============================================================================
! SUBROUTINE: RAN2 - RANDOM NUMBER GENERATOR
! Purpose: Generate random numbers using combined multiplicative congruential method
! Input: idum - seed (negative to initialize)
! Output: x - random number between 0 and 1
!         ix1, ix2, ix3 - internal states
!===============================================================================

	subroutine ran2(idum,x,ix1,ix2,ix3)
	implicit real*8(a-h,o-z) 
	real*8 x
	dimension r(97)
	
	! Parameters for three multiplicative congruential generators
	parameter (m1=259200,ia1=7141,ic1=54773,rm1=3.85802e-6)
	parameter (m2=134456,ia2=8121,ic2=28411,rm2=7.43738e-6)
	parameter (m3=243000,ia3=4561,ic3=51349)
	data iff/0/
	
	! Initialize generators if needed
	if(idum.lt.0.or.iff.eq.0) then
		iff=1
		ix1=mod(ic1-idum,m1)
		ix1=mod(ia1*ix1+ic1,m1)
		ix2=mod(ix1,m2)
		ix1=mod(ia1*ix1+ic1,m1)
		ix3=mod(ix1,m3)
		
		! Fill shuffle table
		do j=1,97
			ix1=mod(ia1*ix1+ic1,m1)
			ix2=mod(ia2*ix2+ic2,m2)
			r(j)=(float(ix1)+float(ix2)*rm2)*rm1
		enddo
		
		idum=1
	endif
	
	! Generate next random number
	ix1=mod(IA1*IX1+IC1,M1)
	ix2=mod(IA2*IX2+IC2,M2)
	ix3=mod(IA3*IX3+IC3,M3) 
	
	! Select from shuffle table
	J=1+(97*IX3)/M3
	IF(J.GT.97.OR.J.LT.1) PAUSE
	
	! Replace table entry and return random number
	R(J)=(FLOAT(IX1)+FLOAT(IX2)*RM2)*RM1
	X=R(J)
	
	RETURN
	END