{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE RankNTypes #-}

module Data.Accelerate.TensorFlow.Desugar where

import Data.Array.Accelerate.AST.Operation
    ( expType,
      paramsIn',
      fromGrounds,
      PrimFun(..),
      PreArgs((:>:), ArgsNil),
      ExpVars,
      PreOpenAcc(..),
      GroundR(GroundRscalar),
      Modifier(In, Mut, Out),
      PreOpenExp(..),
      PreOpenFun(Body, Lam),
      ArrayInstr(..),
      GroundVars,
      OperationAcc,
      Out,
      In,
      Mut,
      Arg(ArgFun, ArgArray, ArgVar), Var (..), PrimConst (..), Fun', PrimMaybe )
import Data.Array.Accelerate.AST.LeftHandSide
    ( LeftHandSide(..) )
import Data.Array.Accelerate.Backend
    ( DesugarAcc(mkGenerate, mkPermute, mkMap) )
import Data.Array.Accelerate.Type
    ( Word8,
      scalarTypeWord8,
      scalarTypeInt,
      IntegralType(..),
      NumType(IntegralNumType, FloatingNumType),
      SingleType(NumSingleType),
      ScalarType(SingleScalarType), BoundedType (..), Int64 )
import Data.Array.Accelerate.Representation.Array
    ( Buffer, Buffers, ArrayR(ArrayR) )
import Data.Array.Accelerate.Representation.Type
    ( Distributes(reprIsSingle),
      Distribute,
      TupR(TupRpair, TupRunit, TupRsingle),
      TypeR )
import Data.Array.Accelerate.Trafo.Var
    ( declareVars, DeclareVars(DeclareVars) )
import Data.Array.Accelerate.AST.Idx ( Idx(ZeroIdx, SuccIdx) )
import Data.Array.Accelerate.Trafo.Operation.Substitution
    ( weakenArrayInstr, aletUnique, Sink(..) )
import Data.Array.Accelerate.AST.Environment
    ( mapEnv, prj', (.>), weakenId, push', type (:>), Env(Empty) )
import Data.Array.Accelerate.Representation.Ground ( buffersR )
import Data.Array.Accelerate.Trafo.Desugar ( desugarAlloc )
import Data.Array.Accelerate.Trafo.Exp.Substitution
    ( weakenVars, SinkExp(weakenE), rebuildNoArrayInstr )
import Data.Type.Equality ( type (:~:)(Refl) )
import Data.Array.Accelerate.Representation.Shape
    ( dim1, shapeType, ShapeR(..), DIM1 )
import Prelude hiding (exp)

import Data.Accelerate.TensorFlow.Type
    ( TensorTypeDict(TensorTypeDict),
      tfTensorTypeDict,
      tfAllDict,
      tfFloatDict,
      tfIntDict,
      tfModDict,
      tfNum'Dict,
      tfNumDict,
      tfOrdDict,
      OneOfDict(OneOfDict), zero, tfTensorTypeDict', tfAllDict' )
import Data.Accelerate.TensorFlow.Operation
    ( TensorOp(..),
      ScatterFun(..) )
import Data.Array.Accelerate.Interpreter (evalMinBound, evalPi, evalMaxBound)
import Data.Array.Accelerate.Representation.Slice
    ( SliceIndex(..), sliceDomainR, sliceEltR, sliceShapeR )

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a))

instance Sink BufferIdx where
  weaken :: (env :> env') -> forall t. BufferIdx env t -> BufferIdx env' t
  weaken w (BIdx idx) = BIdx (weaken w idx)

type BufferEnv benv env = Env (BufferIdx benv) env

instance DesugarAcc TensorOp where
  mkMap :: Arg env (Fun' (s -> t)) -> Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkExp (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate :: Arg env (Fun' (sh -> t)) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
  mkGenerate f (ArgArray _ (ArrayR sh t) gv gvb)
    | DeclareVars lhs w k          <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs' w' k'       <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs'' w'' k''    <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhs''' w''' k''' <- declareVars $ buffersR $ shapeType sh
    = aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
      -- 1) Create a Tensor of flattened shape sh with only ones
      aletUnique lhs' (desugarAlloc (ArrayR dim1 (TupRsingle scalarTypeInt)) 
        (fromGrounds (TupRpair TupRunit (k weakenId)))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        (TConstant scalarTypeInt 1)
        (ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) 
          (TupRpair TupRunit (k w')) (k' weakenId) :>: ArgsNil)
      ) $
      -- 2) Obtain a tensor of indices (tf.where returns list of indices pointing to values > 0)
      aletUnique lhs'' (desugarAlloc (ArrayR dim1 (TupRsingle scalarTypeInt)) 
        (fromGrounds (TupRpair TupRunit (k w')))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (Exec
        TWhere
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) 
          (TupRpair TupRunit (k (w'' .> w'))) (k' w'') :>:
         ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) 
          (TupRpair TupRunit (k (w'' .> w'))) (k'' weakenId) :>:
         ArgsNil)
      ) $
      -- 3) Convert 1d indices to multidimensional indices
      aletUnique lhs''' (desugarAlloc 
        (ArrayR dim1 (shapeType sh)) 
        (fromGrounds $ TupRpair TupRunit (k (w'' .> w')))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkMap
        (ArgFun $ Lam (LeftHandSideSingle scalarTypeInt) $ 
          Body (FromIndex sh 
            (paramsIn' $ fromGrounds $ weakenVars (w''' .> w'' .> w' .> w) gv) 
            (Evar (Var scalarTypeInt ZeroIdx))))
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) 
          (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k'' w'''))
        (ArgArray Out (ArrayR dim1 (shapeType sh)) 
          (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
      )
      -- 4) Apply f to the indices
      (mkMap
        (weaken (w''' .> w'' .> w' .> w) f)
        (ArgArray In (ArrayR dim1 (shapeType sh)) 
          (TupRpair TupRunit (k (w''' .> w'' .> w'))) (k''' weakenId))
        (ArgArray Out 
          (ArrayR dim1 t) 
          (TupRpair TupRunit (k (w''' .> w'' .> w'))) 
          (weakenVars (w''' .> w'' .> w' .> w) gvb))
      )

  mkPermute :: Arg env (Fun' (e -> e -> e)) -> Arg env (Mut sh' e) -> Arg env (Fun' (sh -> PrimMaybe sh')) -> Arg env (In sh e) -> OperationAcc TensorOp env ()
  mkPermute
    (ArgFun comb)
    (ArgArray _ (ArrayR sh' _) gv' gvb')
    perm
    (ArgArray _ (ArrayR sh t) gv gvb)
    | TensorTypeDict                  <- tfTensorTypeDict' t
    , OneOfDict                       <- tfAllDict' t
    , maybeSh'                        <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh'))
    , DeclareVars lhs w k             <- declareVars $ buffersR maybeSh'
    , DeclareVars lhs' w' k'          <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
    , DeclareVars lhs'' w'' k''       <- declareVars $ buffersR (shapeType sh')
    , DeclareVars lhs''' w''' k'''    <- declareVars $ buffersR t
    , DeclareVars lhs'''' w'''' k'''' <- declareVars $ buffersR (TupRsingle scalarTypeInt)
    , DeclareVars lhsSh' _ kSh'       <- declareVars $ shapeType sh'
    , DeclareVars lhsSh'' wSh'' kSh'' <- declareVars $ shapeType sh'
    = -- 1) Create an array of maybeSh' with perm
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
      ) $
      -- 2) To apply boolean mask (filter) with 1D arrays, we need to flatten. Calculate the dim1 size.
      aletUnique lhs' (Compute (ShapeSize sh (paramsIn' $ weakenVars w $ fromGrounds gv))) $
      -- 3) Get 1D array of indices (sh'), by filtering with predicate isJust and map with fromJust.
      aletUnique lhs'' (desugarAlloc (ArrayR sh (shapeType sh')) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask (shapeType sh')
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' w'')) (isJust (TupRpair TupRunit (shapeType sh')) (k (w'' .> w'))))
        (fromJust (shapeType sh') (k (w'' .> w')))
        (k'' weakenId)
      ) $
      -- 4) Get 1D array of updates by filtering with predicate isJust.
      aletUnique lhs''' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit 
      (booleanMask t
        (ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) (TupRpair TupRunit (k' (w''' .> w''))) (isJust (TupRpair TupRunit (shapeType sh')) (k (w''' .> w'' .> w'))))
        (weakenVars (w''' .> w'' .> w' .> w) gvb)
        (k''' weakenId) 
      ) $
      -- 5) Map array of indices (sh') with toIndex to obtain 1D array of flattened indices.
      aletUnique lhs'''' (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (fromGrounds (weakenVars (w''' .> w'' .> w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkMap
        (ArgFun $ Lam lhsSh' $ Body (Let lhsSh'' (paramsIn' $ fromGrounds (weakenVars (w'''' .> w''' .> w'' .> w' .> w) gv')) $ toIndex sh' (kSh'' weakenId) (kSh' wSh'')))
        (ArgArray In (ArrayR dim1 (shapeType sh')) (TupRpair TupRunit (k' (w'''' .> w''' .> w''))) (k'' (w'''' .> w''')))
        (ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k' (w'''' .> w''' .> w''))) (k'''' weakenId))
      ) $
      -- 4) Apply tensor scatter to 1D array of source values, 1D array of flattened indices and 1D array of updates.
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR dim1 t) (TupRpair TupRunit (k' (w'''' .> w''' .> w''))) (weakenVars (w'''' .> w''' .> w'' .> w' .> w) gvb') :>:
        ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (TupRpair TupRunit (k' (w'''' .> w''' .> w''))) (k'''' weakenId) :>:
        ArgArray In (ArrayR dim1 t) (TupRpair TupRunit (k' (w'''' .> w''' .> w''))) (k''' w'''') :>:
        ArgsNil
      )
        where
          isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
          isJust _ (TupRpair word8 _) = word8
          isJust _ _ = error "impossible"

          fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
          fromJust _ (TupRpair _ (TupRpair _ a)) = a
          fromJust _ _ = error "impossible"

          scatterFun :: ScatterFun = case comb of
            Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp fun (Pair (Evar (Var _ (SuccIdx ZeroIdx))) (Evar (Var _ ZeroIdx)))))) -> case fun of
              PrimAdd _ -> ScatterFunAdd -- (+)
              PrimSub _ -> ScatterFunSub -- (-)
              PrimMin _ -> ScatterFunMin -- min
              PrimMax _ -> ScatterFunMax -- max
              _         -> error "only add, sub, min, max, const allowed as combination function for permute not supported"
            Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (Evar (Var _ (SuccIdx ZeroIdx))))) -> ScatterFunUpdate -- const
            _ -> error "only add, sub, min, max, const allowed as combination function for permute not supported"
            
mkExp :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkExp _ (Const s e) aOut | OneOfDict <- tfAllDict s = Exec (TConstant s e) $ aOut :>: ArgsNil

mkExp env (PrimApp f exp) aOut = mkPrimFun f env exp aOut

mkExp env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkExp
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkExp env (Evar (Var st idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          TId
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkExp env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkExp env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

--
mkExp env (ToIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ shapeType sh
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (toIndex sh (k w') (k' weakenId)))) aOut

mkExp env (ShapeSize sh exp) aOut
  | DeclareVars lhs _ k <- declareVars $ shapeType sh
  = mkExp env (Let lhs exp (shapeSize sh (k weakenId))) aOut

mkExp env (FromIndex sh exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ shapeType sh
  , DeclareVars lhs' w' k' <- declareVars $ TupRsingle scalarTypeInt
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (fromIndex sh (k w') (k' weakenId)))) aOut

mkExp env (IndexSlice slix exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceDomainR slix)
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexSlice slix (k w') (k' weakenId)))) aOut

mkExp env (IndexFull slix exp1 exp2) aOut
  | DeclareVars lhs w k <- declareVars $ sliceEltR slix
  , DeclareVars lhs' w' k' <- declareVars $ shapeType (sliceShapeR slix)
  = mkExp env (Let lhs exp1 (Let lhs' (weakenE w exp2) (indexFull slix (k w') (k' weakenId)))) aOut

mkExp env (Cond cond exp1 exp2) (ArgArray _ (ArrayR sh t) gv gvb)
  -- todo add preprocess check to throw an error if theres a loop, mem access, etc.
  | DeclareVars lhs w k <- declareVars $ buffersR $ TupRsingle scalarTypeWord8
  , DeclareVars lhs' w' k' <- declareVars $ buffersR t
  , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
  = aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeWord8)) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w cond) (ArgArray Out (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars w gv) (k weakenId))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars w gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp1) (ArgArray Out (ArrayR sh t) (weakenVars (w' .> w) gv) (k' weakenId))) $
    aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w'' .> w' .> w) env) (weakenArrayInstr (w'' .> w' .> w) exp2) (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId)))
    (select
      t
      (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (k (w'' .> w')))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k' w''))
      (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb))
    )

mkExp _ Nil _ = Return TupRunit

mkExp env (ArrayInstr (Index var) exp) (ArgArray _ (ArrayR sh t@(TupRsingle st)) gv gvb)
  | Refl <- reprIsSingle @ScalarType @t @Buffer st
  , OneOfDict <- tfAllDict st
  , i <- expType exp
  , DeclareVars lhs w k <- declareVars $ TupRsingle $ GroundRscalar scalarTypeInt
  , DeclareVars lhs' w' k' <- declareVars $ buffersR i
  =  -- To apply gather to 1d array, calculate the dim1 size
    aletUnique lhs (Compute (ShapeSize sh (paramsIn' $ fromGrounds gv))) $
    aletUnique lhs' (desugarAlloc (ArrayR sh i) (weakenVars w $ fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv (w' .> w) env) (weakenArrayInstr (w' .> w) exp) (ArgArray Out (ArrayR dim1 i) (TupRpair TupRunit (k w')) (k' weakenId)))
    (Exec
      TGather
      (
        ArgArray In (ArrayR dim1 t) (TupRpair TupRunit (k w')) (TupRsingle (weaken (w' .> w) var)) :>:
        ArgArray In (ArrayR dim1 i) (TupRpair TupRunit (k w')) (k' weakenId) :>:
        ArgArray Out (ArrayR dim1 t) (TupRpair TupRunit (k w')) (weakenVars (w' .> w) gvb) :>:
        ArgsNil
      )
    )

mkExp _ (ArrayInstr (Parameter var@(Var st _)) Nil) aOut
  | OneOfDict <- tfAllDict st
  = Exec
      (TVar st)
      (
        ArgVar (TupRsingle var) :>:
        aOut :>:
        ArgsNil
      )

mkExp env (Undef st) aOut = mkExp env (Const st (zero st)) aOut
mkExp env (Coerce stIn stOut exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | OneOfDict <- tfAllDict stIn
  , OneOfDict <- tfAllDict stOut
  , a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (Exec
      TCast
      ( ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
        ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb) :>:
        ArgsNil)
     )

mkExp env (PrimConst (PrimMinBound bt@(IntegralBoundedType it))) aOut = mkExp env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMinBound bt)) aOut
mkExp env (PrimConst (PrimMaxBound bt@(IntegralBoundedType it))) aOut = mkExp env (Const (SingleScalarType (NumSingleType (IntegralNumType it))) (evalMaxBound bt)) aOut
mkExp env (PrimConst (PrimPi ft)) aOut                                = mkExp env (Const (SingleScalarType (NumSingleType (FloatingNumType ft))) (evalPi ft)) aOut

mkExp env (Foreign _ _ fallback exp) (ArgArray _ (ArrayR sh t) gv gvb)
  | a <- expType exp
  , DeclareVars lhs w k <- declareVars $ buffersR a
  = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (mkExp (weakenEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId)))
    (mkMap
      (ArgFun (rebuildNoArrayInstr fallback))
      (ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId))
      (ArgArray Out (ArrayR sh t) (weakenVars w gv) (weakenVars w gvb)))

-- Not supported
mkExp _ VecPack {} _   = error "VecPack operation not supported by TensorFlow backend."
mkExp _ VecUnpack {} _ = error "VecUnpack operation not supported by TensorFlow backend."
mkExp _ Case {} _      = error "Case operation not supported by TensorFlow backend." -- TODO: check with Ivo
mkExp _ While {} _     = error "While operation not supported by TensorFlow backend."

mkExp _ _ _ = error "impossible"

select :: TypeR t -> Arg env (In sh Word8) -> Arg env (In sh t) -> Arg env (In sh t) -> Arg env (Out sh t) -> PreOpenAcc TensorOp env ()
select (TupRpair t1 t2) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ gvIn2 (TupRpair gvbIn21 gvbIn22)) (ArgArray _ _ gvIn3 (TupRpair gvbIn31 gvbIn32)) (ArgArray _ _ gvOut (TupRpair gvbOut1 gvbOut2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (select t1 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t1) gvIn2 gvbIn21) (ArgArray In (ArrayR sh t1) gvIn3 gvbIn31) (ArgArray Out (ArrayR sh t1) gvOut gvbOut1))
  (select t2 (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray In (ArrayR sh t2) gvIn2 gvbIn22) (ArgArray In (ArrayR sh t2) gvIn3 gvbIn32) (ArgArray Out (ArrayR sh t2) gvOut gvbOut2))
select t@(TupRsingle s) (ArgArray _ (ArrayR sh tWord8) gvIn1 gvbIn1) (ArgArray _ _ _ gvbIn2) (ArgArray _ _ _ gvbIn3) (ArgArray _ _ gvOut gvbOut)
  | OneOfDict <- tfAllDict s
  = Exec
    TSelect
    (ArgArray In (ArrayR sh tWord8) gvIn1 gvbIn1 :>:
     ArgArray In (ArrayR sh (TupRsingle s)) gvIn1 gvbIn2 :>:
     ArgArray In (ArrayR sh (TupRsingle s)) gvIn1 gvbIn3 :>:
     ArgArray Out (ArrayR sh t) gvOut gvbOut :>:
     ArgsNil)
select TupRunit _ _ _ _ = Return TupRunit
select _ _ _ _ _ = error "impossible"

mkPrimFun :: PrimFun (a -> t) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkPrimFun (PrimAdd nt)  | OneOfDict <- tfNumDict nt                   = mkBinaryPrimFun TAdd
mkPrimFun (PrimMul nt)  | OneOfDict <- tfNumDict nt                   = mkBinaryPrimFun TMul
mkPrimFun (PrimSub nt)  | OneOfDict <- tfNumDict nt                   = mkBinaryPrimFun TSub
mkPrimFun (PrimNeg nt)  | OneOfDict <- tfNum'Dict nt                  = mkUnaryPrimFun TNeg
mkPrimFun (PrimAbs nt)  | OneOfDict <- tfNum'Dict nt                  = mkUnaryPrimFun TAbs
mkPrimFun (PrimSig nt)  | OneOfDict <- tfNum'Dict nt                  = mkUnaryPrimFun TSign

mkPrimFun (PrimQuot it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkBinaryPrimFun TTruncateDiv
mkPrimFun (PrimRem it)  | OneOfDict <- tfModDict it                   = mkBinaryPrimFun TTruncateMod
mkPrimFun (PrimQuotRem it)                                            = mkPrimFun2 (PrimQuot it) (PrimRem it)

mkPrimFun (PrimIDiv it) | OneOfDict <- tfNumDict (IntegralNumType it) = mkBinaryPrimFun TRealDiv
mkPrimFun (PrimMod it)  | OneOfDict <- tfModDict it                   = mkBinaryPrimFun TTruncateMod

mkPrimFun (PrimDivMod it)                                             = mkPrimFun2 (PrimIDiv it) (PrimMod it)

mkPrimFun (PrimBAnd it) | OneOfDict <- tfIntDict it                   = mkBinaryPrimFun TBitwiseAnd
mkPrimFun (PrimBOr it)  | OneOfDict <- tfIntDict it                   = mkBinaryPrimFun TBitwiseOr
mkPrimFun (PrimBXor it) | OneOfDict <- tfIntDict it                   = mkBinaryPrimFun TBitwiseXor
mkPrimFun (PrimBNot it) | OneOfDict <- tfIntDict it                   = mkUnaryPrimFun TInvert
mkPrimFun (PrimBShiftL it) | OneOfDict <- tfIntDict it                = undefined
mkPrimFun (PrimBShiftR it) | OneOfDict <- tfIntDict it                = undefined
mkPrimFun (PrimBRotateL it) | OneOfDict <- tfIntDict it               = undefined
mkPrimFun (PrimBRotateR it) | OneOfDict <- tfIntDict it               = undefined
mkPrimFun (PrimPopCount it) | OneOfDict <- tfIntDict it               = undefined
mkPrimFun (PrimCountLeadingZeros it) | OneOfDict <- tfIntDict it      = undefined
mkPrimFun (PrimCountTrailingZeros it) | OneOfDict <- tfIntDict it     = undefined

mkPrimFun (PrimFDiv ft) | OneOfDict <- tfNumDict (FloatingNumType ft) = mkBinaryPrimFun TRealDiv
mkPrimFun (PrimRecip ft) | OneOfDict <- tfFloatDict ft                = mkUnaryPrimFun TReciprocal
mkPrimFun (PrimSin ft) | OneOfDict <- tfFloatDict ft                  = mkUnaryPrimFun TSin
mkPrimFun (PrimCos ft) | OneOfDict <- tfFloatDict ft                  = mkUnaryPrimFun TCos
mkPrimFun (PrimTan ft) | OneOfDict <- tfFloatDict ft                  = mkUnaryPrimFun TTan
mkPrimFun (PrimAsin ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TAsin
mkPrimFun (PrimAcos ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TAcos
mkPrimFun (PrimAtan ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TAtan
mkPrimFun (PrimSinh ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TSinh
mkPrimFun (PrimCosh ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TCosh
mkPrimFun (PrimTanh ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TTanh
mkPrimFun (PrimAsinh ft) | OneOfDict <- tfFloatDict ft                = mkUnaryPrimFun TAsinh
mkPrimFun (PrimAcosh ft) | OneOfDict <- tfFloatDict ft                = mkUnaryPrimFun TAcosh
mkPrimFun (PrimAtanh ft) | OneOfDict <- tfFloatDict ft                = mkUnaryPrimFun TAtanh
mkPrimFun (PrimExpFloating ft) | OneOfDict <- tfFloatDict ft          = mkUnaryPrimFun TExp
mkPrimFun (PrimSqrt ft) | OneOfDict <- tfFloatDict ft                 = mkUnaryPrimFun TSqrt
mkPrimFun (PrimLog ft) | OneOfDict <- tfFloatDict ft                  = mkUnaryPrimFun TLog
mkPrimFun (PrimFPow ft) | OneOfDict <- tfFloatDict ft                 = mkBinaryPrimFun TPow
mkPrimFun (PrimLogBase ft) | OneOfDict <- tfFloatDict ft              = undefined -- mkBinaryPrimFun TLog1p

mkPrimFun (PrimTruncate _ _)                                          = error "Truncate to 0 not supported by TensorFlow."
mkPrimFun (PrimRound ft it) 
  | OneOfDict <- tfFloatDict ft , OneOfDict <- tfIntDict it           = mkUnaryPrimFun TRound
mkPrimFun (PrimFloor ft it)
  | OneOfDict <- tfFloatDict ft , OneOfDict <- tfIntDict it           = mkUnaryPrimFun TFloor
mkPrimFun (PrimCeiling ft it)
  | OneOfDict <- tfFloatDict ft , OneOfDict <- tfIntDict it           = mkUnaryPrimFun TCeil

mkPrimFun (PrimAtan2 ft) | OneOfDict <- tfFloatDict ft                = mkBinaryPrimFun TAtan2
mkPrimFun (PrimIsNaN ft) | OneOfDict <- tfFloatDict ft                = mkUnaryPrimFun TIsNan
mkPrimFun (PrimIsInfinite ft) | OneOfDict <- tfFloatDict ft           = mkUnaryPrimFun TIsInf

mkPrimFun (PrimLt st) | OneOfDict <- tfOrdDict st                     = mkBinaryPrimFun TLess
mkPrimFun (PrimGt st) | OneOfDict <- tfOrdDict st                     = mkBinaryPrimFun TGreater
mkPrimFun (PrimLtEq st) | OneOfDict <- tfOrdDict st                   = mkBinaryPrimFun TLessEqual
mkPrimFun (PrimGtEq st) | OneOfDict <- tfOrdDict st                   = mkBinaryPrimFun TGreaterEqual
mkPrimFun (PrimEq st) | OneOfDict <- tfOrdDict st                     = mkBinaryPrimFun TEqual
mkPrimFun (PrimNEq st) | OneOfDict <- tfOrdDict st                    = mkBinaryPrimFun TNotEqual
mkPrimFun (PrimMax st) | OneOfDict <- tfOrdDict st                    = mkBinaryPrimFun TMaximum
mkPrimFun (PrimMin st) | OneOfDict <- tfOrdDict st                    = mkBinaryPrimFun TMinimum

mkPrimFun PrimLAnd                                                    = mkBinaryPrimFun TLogicalAnd
mkPrimFun PrimLOr                                                     = mkBinaryPrimFun TLogicalOr
mkPrimFun PrimLNot                                                    = mkUnaryPrimFun TLogicalNot

mkPrimFun (PrimFromIntegral it nt)
  | TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType (IntegralNumType it)))
  , TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType nt))
                                                                      = mkUnaryPrimFun TCast
mkPrimFun (PrimToFloating nt ft)
  | TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType nt))
  ,  TensorTypeDict <- tfTensorTypeDict (SingleScalarType (NumSingleType (FloatingNumType ft)))
                                                                      = mkUnaryPrimFun TCast

mkUnaryPrimFun :: TensorOp (In sh a -> Out sh b -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' a -> Arg env (Out sh b) -> OperationAcc TensorOp env ()
mkUnaryPrimFun op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

mkBinaryPrimFun :: TensorOp (In sh a -> In sh b -> Out sh c -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' (a, b) -> Arg env (Out sh c) -> OperationAcc TensorOp env ()
mkBinaryPrimFun op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | t@(TupRpair a b) <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR t
 = let TupRpair k1 k2 = k weakenId in
   aletUnique lhs (desugarAlloc (ArrayR sh t) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh t) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) k1 :>:
      ArgArray In (ArrayR sh b) (weakenVars w gv) k2 :>:
      weaken w aOut :>:
      ArgsNil
    )
mkBinaryPrimFun _ _ _ _ = error "impossible"

mkTernaryPrimFun :: TensorOp (In sh a -> In sh b -> In sh c -> Out sh d -> ()) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' (a, (b, c)) -> Arg env (Out sh d) -> OperationAcc TensorOp env ()
mkTernaryPrimFun op env exp aOut@(ArgArray _ (ArrayR sh _) gv _)
 | t@(TupRpair a (TupRpair b c)) <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR t
 = let TupRpair k1 (TupRpair k2 k3) = k weakenId in
   aletUnique lhs (desugarAlloc (ArrayR sh t) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkExp -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh t) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    op
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) k1 :>:
      ArgArray In (ArrayR sh b) (weakenVars w gv) k2 :>:
      ArgArray In (ArrayR sh c) (weakenVars w gv) k3 :>:
      weaken w aOut :>:
      ArgsNil
    )
mkTernaryPrimFun _ _ _ _ = error "impossible"

mkPrimFun2 :: PrimFun ((a, a) -> a) -> PrimFun ((a, a) -> a) -> BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' (a, a) -> Arg env (Out sh (a, a)) -> OperationAcc TensorOp env ()
mkPrimFun2 fun1 fun2 env exp (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2)) =
  Alet (LeftHandSideWildcard TupRunit) TupRunit
  (mkPrimFun
    fun1
    env
    exp
    (ArgArray Out (ArrayR sh t1) gv gvb1)
  )
  (mkPrimFun
    fun2
    env
    exp
    (ArgArray Out (ArrayR sh t2) gv gvb2)
  )
mkPrimFun2 _ _ _ _ _ = error "impossible"


booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) (ArgArray _ (ArrayR sh tIn1) gv gvbIn1) gvbIn2 gvbOut -- somehow boolean mask is missing in TF bindings
  -- instead, use where + gather
  | OneOfDict <- tfAllDict s
  , DeclareVars lhs w k <- declareVars $ buffersR (TupRsingle scalarTypeInt)
  = aletUnique lhs (desugarAlloc (ArrayR sh (TupRsingle scalarTypeInt)) (fromGrounds gv)) $
    Alet (LeftHandSideWildcard TupRunit) TupRunit
    (Exec
      TWhere
      (ArgArray In (ArrayR dim1 tIn1) (weakenVars w gv) (weakenVars w gvbIn1) :>:
       ArgArray Out (ArrayR dim1 (TupRsingle scalarTypeInt)) (weakenVars w gv) (k weakenId) :>:
       ArgsNil)
    )
    (Exec
      TGather
      (ArgArray In (ArrayR dim1 t) (weakenVars w gv) (weakenVars w gvbIn2) :>:
       ArgArray In (ArrayR dim1 (TupRsingle scalarTypeInt)) (weakenVars w gv) (k weakenId) :>:
       ArgArray Out (ArrayR dim1 t) (weakenVars w gv) (weakenVars w gvbOut) :>:
       ArgsNil)
    )

booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

shapeSize :: ShapeR sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
shapeSize ShapeRz TupRunit                                = Const scalarTypeInt 1
shapeSize (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz))
  = PrimApp (PrimMul (IntegralNumType TypeInt))
      (Pair
        (shapeSize shr sh)
        (Evar sz))
shapeSize _ _                                             = error "impossible"

toIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' sh -> PreOpenExp (ArrayInstr env) env' Int
toIndex ShapeRz TupRunit TupRunit = Const scalarTypeInt 0
toIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRpair ix (TupRsingle i))
  = PrimApp (PrimAdd (IntegralNumType TypeInt))
      (Pair
        (PrimApp (PrimMul (IntegralNumType TypeInt))
          (Pair
            (toIndex shr sh ix)
            (Evar sz)))
        (Evar i))
toIndex _ _ _                     = error "impossible"

fromIndex :: ShapeR sh -> ExpVars env' sh -> ExpVars env' Int -> PreOpenExp (ArrayInstr env) env' sh
fromIndex ShapeRz TupRunit _                                = Nil
fromIndex (ShapeRsnoc shr) (TupRpair sh (TupRsingle sz)) (TupRsingle i)
  | DeclareVars lhs w k <- declareVars $ TupRsingle scalarTypeInt
  = Pair
    (Let
      lhs
      (PrimApp (PrimQuot TypeInt) (Pair (Evar i) (Evar sz)))
      (fromIndex shr (weakenVars w sh) (k weakenId)))
    (case shr of
      ShapeRz -> Evar i
      _       -> PrimApp (PrimRem TypeInt) (Pair (Evar i) (Evar sz))
    )
fromIndex _ _ _              = error "impossible"

indexSlice :: SliceIndex slix t co sh1 -> ExpVars env' slix -> ExpVars env' sh1 -> PreOpenExp (ArrayInstr env) env' t
indexSlice SliceNil _ _ = Nil
indexSlice (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexSlice slix slx sl) (Evar sz)
indexSlice (SliceFixed slix) (TupRpair slx _) (TupRpair sl _) =
  indexSlice slix slx sl
indexSlice _ _ _ = error "impossible"

indexFull :: SliceIndex slix sl co t -> ExpVars env' slix -> ExpVars env' sl -> PreOpenExp (ArrayInstr env) env' t
indexFull SliceNil _ _ = Nil
indexFull (SliceAll slix) (TupRpair slx TupRunit) (TupRpair sl (TupRsingle sz)) =
  Pair (indexFull slix slx sl) (Evar sz)
indexFull (SliceFixed slix) (TupRpair slz (TupRsingle sz)) sl =
  Pair (indexFull slix slz sl) (Evar sz)
indexFull _ _ _ = error "impossible"

weakenEnv :: Sink f => (env :> env') -> Env (f env) a -> Env (f env') a
weakenEnv w = mapEnv (weaken w)

distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"

-- Examples of using the OperationAcc GADT for desugaring

-- add :: forall env sh. 
--        Arg env (In sh Int64) 
--     -> Arg env (In sh Int64) 
--     -> Arg env (Out sh Int64) 
--     -> OperationAcc TensorOp env ()
-- add argIn1 argIn2 argOut =
--   Exec TAdd (argIn1 :>: argIn2 :>: argOut :>: ArgsNil)

-- mapXTimesTwoPlusOne :: forall env sh. Arg env (In sh Int64) 
--   -> Arg env (Out sh Int64) -> OperationAcc TensorOp env ()
-- mapXTimesTwoPlusOne (ArgArray _ arrayR@(ArrayR _ t) gvIn gvbIn) argOut@(ArgArray _ _ _ gvbOut)
--   | DeclareVars lhs  w  k  <- declareVars $ buffersR t -- variable to new array
--   , DeclareVars lhs' w' k' <- declareVars $ buffersR t -- variable to new array
--   = let
--     sInt64 :: ScalarType Int64
--     sInt64 = SingleScalarType (NumSingleType (IntegralNumType TypeInt64))
--     in 
--     -- Allocate new array
--     aletUnique lhs (desugarAlloc arrayR (fromGrounds gvIn)) $
--     Alet (LeftHandSideWildcard TupRunit) TupRunit
--       (Exec -- Fill new array with the number 2
--         (TConstant sInt64 2) 
--         (ArgArray Out arrayR (weakenVars w gvIn) (k weakenId) :>: ArgsNil)
--       ) $
--       Alet (LeftHandSideWildcard TupRunit) TupRunit
--         (Exec -- (*2) Multiply input array with new array
--           TMul
--           (ArgArray In arrayR (weakenVars w gvIn) (weakenVars w gvbIn) :>:
--            ArgArray In arrayR (weakenVars w gvIn) (k weakenId) :>:
--            weaken w argOut :>: 
--            ArgsNil
--           )
--         ) $
--         -- Allocate new array
--         aletUnique lhs' (desugarAlloc arrayR (fromGrounds (weakenVars w gvIn))) $
--           Alet (LeftHandSideWildcard TupRunit) TupRunit
--             (Exec -- Fill new array with the number 1
--               (TConstant sInt64 1) 
--               (ArgArray Out arrayR (weakenVars (w' .> w) gvIn) (k' weakenId) :>:
--                ArgsNil
--               )
--             ) $
--            Exec -- (+1) Add new array to the result array of (*2)
--              TAdd 
--              (ArgArray In arrayR (weakenVars (w' .> w) gvIn) (weakenVars (w' .> w) gvbOut) :>: 
--               ArgArray In arrayR (weakenVars (w' .> w) gvIn) (k' weakenId) :>: 
--               weaken (w' .> w) argOut :>: 
--               ArgsNil
--              )
